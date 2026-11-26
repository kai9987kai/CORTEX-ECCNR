"""FastAPI backend for the CORTEX-ECCNR Interactive Scientific GUI & Live Observatory.

Exposes real-time simulation stepping, FlyWire connectome telemetry, bounded human
interventions (neurogenesis, synaptogenesis, pruning, state-cloned perturbation,
checkpointing/rollback), interactive lineage evolution + 7-strategy fusion comparison,
and background benchmark execution.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .bptt import run_bptt_dtr, train_bptt_steps, train_ewc_finetune_steps
from .connectome import ConnectomeReference, build_synthetic_connectome, load_flywire_subgraph
from .env import ACTION_NAMES, ECCNRWorld, TASKS
from .evolution import (
    LearnedRepairController,
    LineageNode,
    ParentDiagnostics,
    QualityDiversityArchive,
    cross_lineage_exchange,
    diagnose_parent,
)
from .experiments import run_single_condition
from .fusion import fuse_parents
from .persistence import AppendOnlyJournal, load_brain_checkpoint, save_brain_checkpoint
from .substrate import ConnectomeBrain, SubstrateBudgets

STATIC_DIR = Path(__file__).resolve().parent / "static"

FUSION_STRATEGIES = (
    "best_parent",
    "consensus_median",
    "weighted_union",
    "aligned_averaging",
    "circuit_transplantation",
    "hybrid_exact",
    "gated_specialist_routing",
)


class StepRequest(BaseModel):
    n_steps: int = Field(default=1, ge=1, le=25)
    online_plasticity: bool = False
    explore_noise: float = Field(default=0.02, ge=0.0, le=0.5)
    manual_action: Optional[List[float]] = None


class ResetRequest(BaseModel):
    task: str = "navigation"
    seed: int = 42


class BrainSelectRequest(BaseModel):
    preset: str = "founding_flywire"


class InterveneRequest(BaseModel):
    action: str  # neurogenesis | synaptogenesis | prune | perturb_diagnose | save_checkpoint | rollback
    rationale: str = "Interactive GUI researcher intervention"


class MiniEvolutionRequest(BaseModel):
    n_lineages: int = Field(default=3, ge=2, le=6)
    n_generations: int = Field(default=2, ge=2, le=4)
    steps_per_gen: int = Field(default=240, ge=60, le=1200)
    trainer: str = "bptt"
    fusion_strategy: str = "gated_specialist_routing"


class ExperimentLaunchRequest(BaseModel):
    condition: str = "ewc_dtr_finetune"
    total_env_budget: int = Field(default=2400, ge=400, le=24000)
    trainer: str = "bptt"
    seed: int = 101


class StudioState:
    """Thread-safe live state for the CORTEX-ECCNR Observatory."""

    def __init__(self, project_root: Optional[Path] = None):
        self.lock = threading.RLock()
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data" / "flywire"
        self.results_dir = self.project_root / "results"
        self.ckpt_dir = self.project_root / "checkpoints"
        self.journal = AppendOnlyJournal(self.ckpt_dir / "interventions.jsonl")

        cache_manifest = self.data_dir / "cache" / "manifest.json"
        if cache_manifest.exists():
            self.ref: ConnectomeReference = load_flywire_subgraph(self.data_dir, n_nodes=256, seed=42)
        else:
            self.ref = build_synthetic_connectome(n_nodes=96, seed=42)

        self.budgets = SubstrateBudgets()
        self.base_weights = self.ref.to_initial_weights(row_norm_cap=self.budgets.row_norm_cap)
        self.env = ECCNRWorld(max_steps=80, seed=42)
        self.obs = self.env.reset(task="navigation", seed=42)

        self.brains: Dict[str, ConnectomeBrain] = {}
        self._init_preset_brains()
        self.active_preset = "founding_flywire"
        self.saved_snapshot: Dict[str, Any] = self.active_brain.snapshot()
        self.causal_sensitivity = np.zeros(self.active_brain.max_total_nodes, dtype=np.float32)

        self.reward_history: List[float] = []
        self.pred_err_history: List[float] = []
        self.last_action = np.zeros(5, dtype=np.float32)
        self.last_pred = np.zeros(9, dtype=np.float32)

        # Evolution & fusion studio cache
        self.dag_nodes: List[Dict[str, Any]] = []
        self.qd_cells: List[Dict[str, Any]] = []
        self.fusion_comparisons: List[Dict[str, Any]] = []

        # Background job runner
        self.job_state: Dict[str, Any] = {
            "running": False,
            "progress": 0.0,
            "logs": [],
            "last_result": None,
        }

    def _init_preset_brains(self) -> None:
        b_found = ConnectomeBrain(
            self.ref, lineage_id="FLYWIRE_BASE", generation=0, seed=42,
            budgets=self.budgets, shared_base_weights=self.base_weights,
        )
        b_delay = ConnectomeBrain(
            self.ref, lineage_id="DENDRITIC_DELAY", generation=1, seed=101,
            budgets=self.budgets, shared_base_weights=self.base_weights,
            enable_delays=True, enable_dendritic_gating=True,
        )
        b_delay.add_neuron(role="central", sign=1, birth_event="preset_init", hypothesis="memory_bridge")
        b_delay.add_edge(0, b_delay.n_active - 1, birth_event="preset_init")

        p2 = ConnectomeBrain(
            self.ref, lineage_id="SPEC_OBSTACLE", generation=1, seed=202,
            budgets=self.budgets, shared_base_weights=self.base_weights,
            enable_delays=True, enable_dendritic_gating=True,
        )
        p2.add_neuron(role="motor", sign=-1, birth_event="preset_init", hypothesis="obstacle_reflex")
        d1 = diagnose_parent(b_delay, seeds=[42], max_steps=20)
        d2 = diagnose_parent(p2, seeds=[43], max_steps=20)
        b_hybrid, _, _ = fuse_parents([b_delay, p2], [d1, d2], strategy="hybrid_exact", seed=303)
        b_gated, _, _ = fuse_parents([b_delay, p2], [d1, d2], strategy="gated_specialist_routing", seed=404)

        self.brains = {
            "founding_flywire": b_found,
            "dendritic_delay_specialist": b_delay,
            "hybrid_exact_fused": b_hybrid,
            "gated_specialist_router": b_gated,
        }

    @property
    def active_brain(self) -> ConnectomeBrain:
        return self.brains[self.active_preset]

    def world_telemetry(self) -> Dict[str, Any]:
        goal = self.env.target if (self.env.task != "memory_cue" or self.env.cue_code == 0) else self.env.decoy
        return {
            "task": self.env.task,
            "t": self.env.t,
            "max_steps": self.env.max_steps,
            "x": round(self.env.x, 4),
            "y": round(self.env.y, 4),
            "th": round(self.env.th, 4),
            "v": round(self.env.v, 4),
            "w": round(self.env.w, 4),
            "reach": round(self.env.reach, 4),
            "height": round(self.env.height, 4),
            "grip": round(self.env.grip, 4),
            "energy": round(self.env.energy, 4),
            "stamina": round(self.env.stamina, 4),
            "target": [round(self.env.target[0], 3), round(self.env.target[1], 3)],
            "decoy": [round(self.env.decoy[0], 3), round(self.env.decoy[1], 3)],
            "active_goal": [round(goal[0], 3), round(goal[1], 3)],
            "gate_pos": [round(self.env.gate_pos[0], 3), round(self.env.gate_pos[1], 3)],
            "gate_triggered": self.env.gate_triggered,
            "cue_code": self.env.cue_code,
            "obstacles": self.env.obstacles,
            "resources": self.env.resources,
            "cumulative_reward": round(self.env.cumulative_reward, 4),
            "last_reward": round(self.env.last_reward, 4),
            "collisions": self.env.collisions,
            "subgoals_completed": self.env.subgoals_completed,
            "retina": [round(float(x), 3) for x in self.obs[:135]],
            "last_action": {ACTION_NAMES[i]: round(float(self.last_action[i]), 4) for i in range(5)},
            "reward_history": [round(r, 4) for r in self.reward_history[-80:]],
            "pred_err_history": [round(e, 4) for e in self.pred_err_history[-80:]],
        }


def create_app(project_root: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="CORTEX-ECCNR Scientific Observatory", version="0.2.0")
    studio = StudioState(project_root=project_root)
    app.state.studio = studio

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        with studio.lock:
            reports = sorted(
                p.stem.replace("_report", "")
                for p in studio.results_dir.glob("*_report.json")
            )
            ckpts = sorted(p.stem for p in studio.ckpt_dir.glob("*.json"))
            b = studio.active_brain
            return {
                "dataset_id": studio.ref.dataset_id,
                "provenance_tier": studio.ref.provenance_tier,
                "topology_hash": studio.ref.topology_hash(),
                "n_bio_nodes": studio.ref.n_nodes,
                "n_bio_edges": studio.ref.n_edges,
                "active_preset": studio.active_preset,
                "available_presets": list(studio.brains.keys()),
                "tasks": list(TASKS),
                "fusion_strategies": list(FUSION_STRATEGIES),
                "available_reports": reports,
                "checkpoints": ckpts,
                "brain_summary": {
                    "lineage_id": b.lineage_id,
                    "generation": b.generation,
                    "n_active": b.n_active,
                    "n_novel_neurons": b.overlay.n_novel_neurons,
                    "n_novel_edges": len(b.overlay.novel_edges),
                    "n_pruned_edges": len(b.overlay.pruned_bio_edges),
                    "enable_delays": b.enable_delays,
                    "enable_dendritic_gating": b.enable_dendritic_gating,
                    "has_specialist_router": bool(b.specialist_w_outs and b.router_w is not None),
                    "routing_weights": [round(float(x), 4) for x in b.last_routing_weights],
                    "neuromod": [round(float(x), 4) for x in b.neuromod],
                    "astrocyte_ca": round(float(b.astrocyte_ca), 4),
                    "weight_hash": b.weight_hash(),
                    "energy_used": round(float(b.energy_used), 4),
                },
                "world": studio.world_telemetry(),
                "job": studio.job_state,
            }

    @app.get("/api/graph")
    def get_graph() -> Dict[str, Any]:
        with studio.lock:
            g = studio.active_brain.export_graph_telemetry(max_bio_edges=300)
            for node in g["nodes"]:
                idx = node["idx"]
                node["causal_sensitivity"] = round(float(studio.causal_sensitivity[idx]), 4) if idx < len(studio.causal_sensitivity) else 0.0
            return g

    @app.post("/api/world/reset")
    def reset_world(req: ResetRequest) -> Dict[str, Any]:
        with studio.lock:
            if req.task not in TASKS:
                raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
            studio.obs = studio.env.reset(task=req.task, seed=req.seed)
            studio.active_brain.reset_state()
            studio.reward_history.clear()
            studio.pred_err_history.clear()
            return {"world": studio.world_telemetry()}

    @app.post("/api/world/step")
    def step_world(req: StepRequest) -> Dict[str, Any]:
        with studio.lock:
            b = studio.active_brain
            for _ in range(req.n_steps):
                if studio.env.t >= studio.env.max_steps:
                    studio.obs = studio.env.reset(task=studio.env.task, seed=int(time.time() * 1000) % 10000)
                    b.reset_state()
                act, pred = b.forward(studio.obs, explore_noise=req.explore_noise if req.online_plasticity else 0.0)
                if req.manual_action is not None and len(req.manual_action) == 5:
                    act = np.clip(np.asarray(req.manual_action, dtype=np.float32), -1.0, 1.0)
                res = studio.env.step(act)
                if req.online_plasticity:
                    perr = b.update_plasticity(studio.obs, act, pred, res.obs, res.reward, lr=0.02, update_recurrent=True)
                else:
                    target = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
                    perr = float(np.mean((target - pred) ** 2))
                studio.last_action = act
                studio.last_pred = pred
                studio.reward_history.append(float(res.reward))
                studio.pred_err_history.append(float(perr))
                studio.obs = res.obs
                if res.done:
                    break
            return {
                "world": studio.world_telemetry(),
                "rates": [round(float(x), 4) for x in b.rates[: b.n_active]],
                "neuromod": [round(float(x), 4) for x in b.neuromod],
                "astrocyte_ca": round(float(b.astrocyte_ca), 4),
                "routing_weights": [round(float(x), 4) for x in b.last_routing_weights],
                "energy_used": round(float(b.energy_used), 4),
            }

    @app.post("/api/brain/select")
    def select_brain(req: BrainSelectRequest) -> Dict[str, Any]:
        with studio.lock:
            if req.preset in studio.brains:
                studio.active_preset = req.preset
            else:
                ckpt_meta = studio.ckpt_dir / f"{req.preset}.json"
                if not ckpt_meta.exists():
                    raise HTTPException(status_code=404, detail=f"Unknown preset or checkpoint: {req.preset}")
                loaded = load_brain_checkpoint(ckpt_meta, studio.ref, budgets=studio.budgets)
                studio.brains[req.preset] = loaded
                studio.active_preset = req.preset
            studio.saved_snapshot = studio.active_brain.snapshot()
            return get_status()

    @app.post("/api/brain/intervene")
    def intervene_brain(req: InterveneRequest) -> Dict[str, Any]:
        with studio.lock:
            b = studio.active_brain
            msg = ""
            if req.action == "neurogenesis":
                nid = b.add_neuron(role="central", sign=1, birth_event="gui_human_neurogenesis", hypothesis=req.rationale[:60])
                if nid is None:
                    msg = "Neurogenesis budget cap reached"
                else:
                    top_act = int(np.argmax(np.abs(b.rates[: nid]))) if nid > 0 else 0
                    b.add_edge(nid, top_act, birth_event="gui_wire_in")
                    b.add_edge(0, nid, birth_event="gui_wire_out")
                    msg = f"Spawned novel interneuron #{nid} ({b.node_identity(nid)})"
            elif req.action == "synaptogenesis":
                n_act = b.n_active
                order = np.argsort(np.abs(b.rates[:n_act]))[::-1]
                post, pre = int(order[0]), int(order[min(1, n_act - 1)])
                if post == pre:
                    pre = (post + 1) % n_act
                inn_id = b.add_edge(post, pre, birth_event="gui_human_synaptogenesis", hypothesis=req.rationale[:60])
                msg = f"Created Dale's-law synapse {pre}->{post} ({inn_id})" if inn_id else "Synaptogenesis cap reached"
            elif req.action == "prune":
                n_pruned = b.prune_weak_edges(threshold=0.02, max_prune=8)
                msg = f"Pruned {n_pruned} low-magnitude synapses (reference root IDs preserved)"
            elif req.action == "perturb_diagnose":
                diag = diagnose_parent(b, seeds=[42], max_steps=25)
                studio.causal_sensitivity[: len(diag.causal_node_sensitivity)] = diag.causal_node_sensitivity
                msg = f"Counterfactual perturbation complete: mean reward={diag.mean_reward:.3f}, max node sensitivity={float(np.max(diag.causal_node_sensitivity)):.4f}"
            elif req.action == "save_checkpoint":
                studio.saved_snapshot = b.snapshot()
                manifest = save_brain_checkpoint(b, studio.ckpt_dir, tag="gui", extra_meta={"rationale": req.rationale})
                msg = f"Saved verified checkpoint {manifest['checkpoint_id']} (SHA-256 {manifest['npz_sha256'][:12]}...)"
            elif req.action == "rollback":
                b.restore(studio.saved_snapshot)
                msg = f"Rolled back brain {b.lineage_id} to snapshot hash {b.weight_hash()}"
            else:
                raise HTTPException(status_code=400, detail=f"Unknown intervention: {req.action}")

            studio.journal.log_intervention(
                checkpoint_id=f"{b.lineage_id}_g{b.generation}_{b.weight_hash()}",
                observation_summary={"energy": round(studio.env.energy, 3), "t": float(studio.env.t)},
                prediction_summary={"pred_reward": round(float(studio.last_pred[8]), 4)},
                automatic_judgement="interactive_session",
                supervisor_source="human_gui",
                supervisor_decision="rollback" if req.action == "rollback" else "bounded_edit",
                rationale=req.rationale,
                intervention_applied=msg,
            )
            return {"message": msg, "status": get_status(), "graph": get_graph()}

    @app.post("/api/evolution/mini_run")
    def run_mini_evolution(req: MiniEvolutionRequest) -> Dict[str, Any]:
        with studio.lock:
            controller = LearnedRepairController(seed=77)
            qd = QualityDiversityArchive()
            lineages = [
                ConnectomeBrain(
                    studio.ref, lineage_id=f"L{i}", generation=0, seed=100 + i * 17,
                    budgets=studio.budgets, shared_base_weights=studio.base_weights,
                    enable_delays=True, enable_dendritic_gating=True,
                )
                for i in range(req.n_lineages)
            ]
            dag: List[Dict[str, Any]] = []
            last_diags: List[ParentDiagnostics] = []
            train_seeds = [11, 12, 13]
            val_seeds = [21, 22]

            for g in range(req.n_generations):
                next_gen: List[ConnectomeBrain] = []
                last_diags = []
                for i, p in enumerate(lineages):
                    if req.trainer == "bptt":
                        train_bptt_steps(p, train_seeds, steps=req.steps_per_gen, max_episode_steps=40, update_recurrent=True, structural_plasticity=True)
                    else:
                        from .experiments import train_brain_steps
                        train_brain_steps(p, train_seeds, steps=req.steps_per_gen, max_episode_steps=40, update_recurrent=True, structural_plasticity=True)
                    diag = diagnose_parent(p, val_seeds[:1], max_steps=35)
                    last_diags.append(diag)
                    qd.consider(p, diag)

                    prop = controller.propose(diag, mode="ucb_learned")
                    if g < req.n_generations - 1:
                        child = p.clone_child(f"L{i}", g + 1, seed=300 + g * 20 + i)
                        controller.apply_repair(child, diag, prop)
                        if len(qd.cells) >= 2:
                            donor = next((el for el in qd.cells.values() if el.lineage_id != p.lineage_id), None)
                            if donor:
                                cross_lineage_exchange(child, donor.snapshot, donor.diagnostics, max_graft_edges=2)
                        c_diag = diagnose_parent(child, val_seeds[:1], max_steps=30)
                        delta = c_diag.mean_reward - diag.mean_reward
                        controller.observe_outcome(diag, prop, delta)
                        dag.append({
                            "node_id": f"L{i}_g{g}",
                            "lineage_id": f"L{i}",
                            "generation": g,
                            "parent_id": f"L{i}_g{g-1}" if g > 0 else None,
                            "repair_kind": prop.edit_kind,
                            "target_task": prop.target_task,
                            "predicted_delta": round(prop.predicted_delta, 4),
                            "actual_delta": round(delta, 4),
                            "mean_reward": round(diag.mean_reward, 4),
                            "weight_hash": p.weight_hash(),
                        })
                        next_gen.append(child)
                    else:
                        dag.append({
                            "node_id": f"L{i}_g{g}",
                            "lineage_id": f"L{i}",
                            "generation": g,
                            "parent_id": f"L{i}_g{g-1}" if g > 0 else None,
                            "repair_kind": "terminal_parent",
                            "target_task": prop.target_task,
                            "predicted_delta": 0.0,
                            "actual_delta": 0.0,
                            "mean_reward": round(diag.mean_reward, 4),
                            "weight_hash": p.weight_hash(),
                        })
                if g < req.n_generations - 1:
                    lineages = next_gen

            # Compare all 7 fusion strategies side-by-side!
            comparisons: List[Dict[str, Any]] = []
            promoted_brain: Optional[ConnectomeBrain] = None
            promoted_maps: Dict[int, Dict[int, float]] = {}
            for strat in FUSION_STRATEGIES:
                f_brain, prov, pmaps = fuse_parents(lineages, last_diags, strategy=strat, seed=700)
                f_diag = diagnose_parent(f_brain, val_seeds[:1], max_steps=35)
                comparisons.append({
                    "strategy": strat,
                    "pre_adapt_reward": round(f_diag.mean_reward, 4),
                    "exact_inheritance": prov.exact_inheritance_verified,
                    "weight_hash": prov.inherited_weight_hash,
                    "homologous_matches": prov.homologous_matches,
                    "novel_matches": prov.novel_matches,
                    "unmatched_kept": prov.unmatched_novel_kept,
                    "conflicts": prov.conflicts_recorded,
                })
                if strat == req.fusion_strategy:
                    promoted_brain = f_brain
                    promoted_maps = pmaps

            if promoted_brain is not None:
                # Run rapid EWC-protected DTR consolidation on the chosen fused brain!
                rep_info = run_bptt_dtr(
                    promoted_brain, last_diags, promoted_maps,
                    mode="ewc_staged", steps=120, max_episode_steps=35,
                    prioritize=True, estimate_fisher=True,
                )
                train_ewc_finetune_steps(
                    promoted_brain, train_seeds, steps=120,
                    fisher_diag=rep_info.get("fisher_diag"),
                    ewc_lambda=30.0, rec_lr_scale=0.10, max_episode_steps=35,
                )
                preset_key = f"evolved_{req.fusion_strategy}"
                studio.brains[preset_key] = promoted_brain
                studio.active_preset = preset_key
                studio.saved_snapshot = promoted_brain.snapshot()

            studio.dag_nodes = dag
            studio.qd_cells = qd.summary()
            studio.fusion_comparisons = comparisons
            return {
                "dag": dag,
                "qd_archive": studio.qd_cells,
                "fusion_comparisons": comparisons,
                "promoted_preset": studio.active_preset,
                "status": get_status(),
            }

    @app.get("/api/evolution/state")
    def get_evolution_state() -> Dict[str, Any]:
        with studio.lock:
            return {
                "dag": studio.dag_nodes,
                "qd_archive": studio.qd_cells,
                "fusion_comparisons": studio.fusion_comparisons,
            }

    @app.get("/api/results/{report_name}")
    def get_report(report_name: str) -> Dict[str, Any]:
        safe = "".join(c for c in report_name if c.isalnum() or c == "_")
        json_path = studio.results_dir / f"{safe}_report.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Report not found: {safe}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        csv_path = studio.results_dir / f"{safe}_raw_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            payload["raw_rows"] = df.to_dict(orient="records")
        return payload

    @app.get("/api/journal")
    def get_journal() -> Dict[str, Any]:
        return {"entries": studio.journal.read_all(limit=100)}

    @app.post("/api/experiments/launch")
    def launch_background_experiment(req: ExperimentLaunchRequest) -> Dict[str, Any]:
        with studio.lock:
            if studio.job_state["running"]:
                raise HTTPException(status_code=409, detail="An experiment job is already running")
            studio.job_state = {
                "running": True,
                "progress": 0.1,
                "logs": [f"Starting {req.condition} (budget={req.total_env_budget}, trainer={req.trainer}, seed={req.seed})..."],
                "last_result": None,
            }

        def _worker() -> None:
            try:
                res = run_single_condition(
                    condition=req.condition,
                    ref=studio.ref,
                    train_seeds=[req.seed + 1, req.seed + 2, req.seed + 3],
                    val_seeds=[req.seed + 10, req.seed + 11],
                    sealed_seeds=[9001, 9002, 9003],
                    total_env_budget=req.total_env_budget,
                    n_lineages=3,
                    n_generations=2,
                    max_episode_steps=50,
                    seed=req.seed,
                    trainer=req.trainer,
                )
                with studio.lock:
                    studio.job_state["progress"] = 1.0
                    studio.job_state["logs"].append(
                        f"Completed {req.condition}: sealed_reward={res['sealed_mean_reward']:+.4f}, "
                        f"success={res['sealed_success_rate']*100:.1f}%, wall={res['wall_seconds']}s"
                    )
                    studio.job_state["last_result"] = res
                    studio.job_state["running"] = False
            except Exception as exc:
                with studio.lock:
                    studio.job_state["logs"].append(f"ERROR: {exc}")
                    studio.job_state["running"] = False

        threading.Thread(target=_worker, daemon=True).start()
        return studio.job_state

    @app.get("/api/experiments/job")
    def get_job() -> Dict[str, Any]:
        with studio.lock:
            return studio.job_state

    return app
