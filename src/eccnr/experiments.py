"""Matched-budget experimental matrix, lineage runner, and successive-action population study."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bptt import run_bptt_dtr, train_ann_bptt_steps, train_bptt_steps, train_ewc_finetune_steps
from .connectome import ConnectomeReference, rewire_connectome_degree_preserving
from .env import ECCNRWorld, TASKS
from .evolution import (
    LearnedRepairController,
    LineageNode,
    ParentDiagnostics,
    QualityDiversityArchive,
    cross_lineage_exchange,
    diagnose_parent,
    run_episode_and_capture,
)
from .fusion import ProvenanceReport, fuse_parents
from .replay import (
    StageMetrics,
    evaluate_brain,
    run_adaptation_stage,
    run_autonomous_finetune,
    run_developmental_trace_replay,
)
from .substrate import ConnectomeBrain, RecurrentANNAnalogue, SubstrateBudgets


def train_brain_steps(
    brain: ConnectomeBrain,
    train_seeds: List[int],
    steps: int,
    max_episode_steps: int = 80,
    lr: float = 0.02,
    update_recurrent: bool = True,
    structural_plasticity: bool = False,
) -> int:
    """Train a brain for an exact environment step budget."""
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])
    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        obs = env.reset(task=task, seed=seed)
        brain.reset_state()
        for t in range(max_episode_steps):
            act, pred = brain.forward(obs, explore_noise=0.05)
            res = env.step(act)
            brain.update_plasticity(
                obs=obs,
                action_taken=act,
                pred=pred,
                next_obs=res.obs,
                reward=res.reward,
                lr=lr,
                update_recurrent=update_recurrent,
            )
            used += 1
            obs = res.obs
            if res.done or used >= steps:
                break
        if structural_plasticity and (ep_idx % 4 == 3):
            top_act = np.argsort(np.abs(brain.rates[: brain.n_active]))[::-1]
            if len(top_act) >= 2:
                brain.add_edge(int(top_act[0]), int(top_act[1]), birth_event="online_hebbian_synaptogenesis")
            brain.prune_weak_edges(threshold=0.008, max_prune=4)
        ep_idx += 1
    return used


def run_single_condition(
    condition: str,
    ref: ConnectomeReference,
    train_seeds: List[int],
    val_seeds: List[int],
    sealed_seeds: List[int],
    total_env_budget: int = 4800,
    n_lineages: int = 4,
    n_generations: int = 3,
    max_episode_steps: int = 80,
    seed: int = 42,
    trainer: str = "eligibility",
) -> Dict[str, Any]:
    """Run one condition from the ECCNR experimental matrix under a matched env-step budget."""
    t0 = time.perf_counter()
    budgets = SubstrateBudgets()
    rng = np.random.default_rng(seed)

    def _train_brain(b: ConnectomeBrain, s_budget: int, update_rec: bool = True, struct_pl: bool = False) -> int:
        if trainer == "bptt":
            return train_bptt_steps(
                b, train_seeds, s_budget, max_episode_steps=max_episode_steps,
                update_recurrent=update_rec, structural_plasticity=struct_pl,
            )
        return train_brain_steps(
            b, train_seeds, s_budget, max_episode_steps=max_episode_steps,
            update_recurrent=update_rec, structural_plasticity=struct_pl,
        )

    if condition == "ann_analogue":
        ann = RecurrentANNAnalogue(hidden_dim=ref.n_nodes, seed=seed)
        env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])
        if trainer == "bptt":
            used = train_ann_bptt_steps(ann, train_seeds, total_env_budget, max_episode_steps=max_episode_steps)
        else:
            used = 0
            ep_idx = 0
            while used < total_env_budget:
                task = TASKS[ep_idx % len(TASKS)]
                s = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
                obs = env.reset(task=task, seed=s)
                ann.reset_state()
                for _ in range(max_episode_steps):
                    act, pred = ann.forward(obs, explore_noise=0.05)
                    res = env.step(act)
                    ann.update_plasticity(obs, act, pred, res.obs, res.reward, lr=0.02)
                    used += 1
                    obs = res.obs
                    if res.done or used >= total_env_budget:
                        break
                ep_idx += 1

        # Sealed evaluation for ANN
        per_task_rew: Dict[str, List[float]] = {t: [] for t in TASKS}
        per_task_succ: Dict[str, List[float]] = {t: [] for t in TASKS}
        perrs: List[float] = []
        for task in TASKS:
            for s in sealed_seeds:
                obs = env.reset(task=task, seed=s)
                ann.reset_state()
                tot = 0.0
                succ = False
                for _ in range(max_episode_steps):
                    act, pred = ann.forward(obs, explore_noise=0.0)
                    res = env.step(act)
                    tot += res.reward
                    target = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
                    perrs.append(float(np.mean((target - pred) ** 2)))
                    obs = res.obs
                    if res.done:
                        succ = bool(res.info.get("success", False))
                        break
                per_task_rew[task].append(tot)
                per_task_succ[task].append(1.0 if succ else 0.0)

        return {
            "condition": condition,
            "seed": seed,
            "env_steps_used": used,
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "energy_used": round(ann.energy_used, 4),
            "sealed_mean_reward": float(np.mean([np.mean(v) for v in per_task_rew.values()])),
            "sealed_success_rate": float(np.mean([np.mean(v) for v in per_task_succ.values()])),
            "sealed_pred_err": float(np.mean(perrs)),
            "per_task_reward": {t: float(np.mean(v)) for t, v in per_task_rew.items()},
            "exact_inheritance_verified": False,
        }

    # Connectome-based conditions
    active_ref = rewire_connectome_degree_preserving(ref, seed=seed + 999) if condition == "rewired_control" else ref
    base_weights = active_ref.to_initial_weights(row_norm_cap=budgets.row_norm_cap)

    if condition == "fixed_connectome":
        brain = ConnectomeBrain(active_ref, lineage_id="FIXED", generation=0, seed=seed, budgets=budgets, shared_base_weights=base_weights)
        # Only readout training, topology and recurrent weights fixed
        used = _train_brain(brain, total_env_budget, update_rec=False, struct_pl=False)
        metrics = evaluate_brain(brain, sealed_seeds, stage="sealed", max_steps=max_episode_steps)
        return _format_result(condition, seed, used, t0, brain, metrics)

    if condition == "single_plastic":
        brain = ConnectomeBrain(active_ref, lineage_id="SINGLE", generation=0, seed=seed, budgets=budgets, shared_base_weights=base_weights)
        used = _train_brain(brain, total_env_budget, update_rec=True, struct_pl=True)
        metrics = evaluate_brain(brain, sealed_seeds, stage="sealed", max_steps=max_episode_steps)
        return _format_result(condition, seed, used, t0, brain, metrics)

    if condition == "sequential_correction":
        # 1 lineage x (n_lineages * n_generations) sequential generations
        total_gens = n_lineages * n_generations
        steps_per_gen = max(80, total_env_budget // total_gens)
        controller = LearnedRepairController(seed=seed)
        brain = ConnectomeBrain(active_ref, lineage_id="SEQ", generation=0, seed=seed, budgets=budgets, shared_base_weights=base_weights)
        used = 0
        for g in range(total_gens):
            used += _train_brain(brain, steps_per_gen, update_rec=True, struct_pl=True)
            diag = diagnose_parent(brain, val_seeds[:2], max_steps=max_episode_steps)
            prop = controller.propose(diag, mode="learned")
            child = brain.clone_child("SEQ", g + 1, seed=seed + g + 1)
            controller.apply_repair(child, diag, prop)
            brain = child
        metrics = evaluate_brain(brain, sealed_seeds, stage="sealed", max_steps=max_episode_steps)
        return _format_result(condition, seed, used, t0, brain, metrics)

    # Multi-lineage evolutionary conditions:
    # Reserve budget for post-fusion adaptation + replay + finetune so total env steps match single_plastic!
    has_fusion = condition not in ("parallel_best",)
    is_advanced = condition in ("ewc_dtr_finetune", "gated_fusion_ewc")
    adapt_budget = int(0.10 * total_env_budget) if has_fusion else 0
    replay_budget = int(0.12 * total_env_budget) if condition in (
        "fusion_ordinary_replay", "fusion_activity_replay", "full_dtr_finetune",
        "rewired_control", "ewc_dtr_finetune", "gated_fusion_ewc",
    ) else 0
    finetune_budget = int(0.10 * total_env_budget) if condition in (
        "full_dtr_finetune", "rewired_control", "ewc_dtr_finetune", "gated_fusion_ewc",
    ) else 0
    evo_budget = total_env_budget - adapt_budget - replay_budget - finetune_budget
    steps_per_lin_gen = max(80, evo_budget // (n_lineages * n_generations))

    controller = LearnedRepairController(seed=seed)
    qd_archive = QualityDiversityArchive()
    lineages: List[ConnectomeBrain] = [
        ConnectomeBrain(
            active_ref, lineage_id=f"L{i}", generation=0, seed=seed + i * 31,
            budgets=budgets, shared_base_weights=base_weights,
            enable_delays=(condition == "gated_fusion_ewc"),
            enable_dendritic_gating=(condition == "gated_fusion_ewc"),
        )
        for i in range(n_lineages)
    ]
    dag: List[LineageNode] = []
    used = 0
    last_diags: List[ParentDiagnostics] = []
    repair_mode = "ucb_learned" if is_advanced else "learned"

    for g in range(n_generations):
        new_lineages: List[ConnectomeBrain] = []
        last_diags = []
        for i, parent in enumerate(lineages):
            used += _train_brain(parent, steps_per_lin_gen, update_rec=True, struct_pl=True)
            diag = diagnose_parent(parent, val_seeds[:2], max_steps=max_episode_steps)
            last_diags.append(diag)
            qd_archive.consider(parent, diag)

            if g < n_generations - 1:
                prop = controller.propose(diag, mode=repair_mode)
                child = parent.clone_child(f"L{i}", g + 1, seed=seed + (g + 1) * 100 + i)
                controller.apply_repair(child, diag, prop)
                if condition == "gated_fusion_ewc" and len(qd_archive.cells) >= 2:
                    # Cross-lineage specialist motif exchange from QD archive
                    donor = next(
                        (el for el in qd_archive.cells.values() if el.lineage_id != parent.lineage_id),
                        None,
                    )
                    if donor is not None:
                        cross_lineage_exchange(child, donor.snapshot, donor.diagnostics, max_graft_edges=3)
                # Quick 1-episode check to score repair prediction
                child_diag = diagnose_parent(child, val_seeds[:1], max_steps=max_episode_steps)
                actual_delta = child_diag.mean_reward - diag.mean_reward
                controller.observe_outcome(diag, prop, actual_delta)
                dag.append(
                    LineageNode(
                        node_id=f"L{i}_g{g}",
                        lineage_id=f"L{i}",
                        generation=g,
                        parent_ids=[f"L{i}_g{g-1}"] if g > 0 else [],
                        repair_kind=prop.edit_kind,
                        predicted_fix_delta=prop.predicted_delta,
                        actual_fix_delta=actual_delta,
                        mean_reward=diag.mean_reward,
                        success_rate=diag.success_rate,
                        weight_hash=parent.weight_hash(),
                        novelty_score=float(np.mean([t.novelty_score for t in diag.traces])) if diag.traces else 0.0,
                    )
                )
                new_lineages.append(child)
            else:
                dag.append(
                    LineageNode(
                        node_id=f"L{i}_g{g}",
                        lineage_id=f"L{i}",
                        generation=g,
                        parent_ids=[f"L{i}_g{g-1}"] if g > 0 else [],
                        repair_kind="terminal_parent",
                        predicted_fix_delta=0.0,
                        actual_fix_delta=0.0,
                        mean_reward=diag.mean_reward,
                        success_rate=diag.success_rate,
                        weight_hash=parent.weight_hash(),
                        novelty_score=float(np.mean([t.novelty_score for t in diag.traces])) if diag.traces else 0.0,
                    )
                )
        if g < n_generations - 1:
            lineages = new_lineages

    if condition == "parallel_best":
        best_idx = int(np.argmax([d.mean_reward for d in last_diags]))
        final_brain = lineages[best_idx]
        metrics = evaluate_brain(final_brain, sealed_seeds, stage="sealed", max_steps=max_episode_steps)
        res = _format_result(condition, seed, used, t0, final_brain, metrics)
        res["dag_nodes"] = len(dag)
        res["qd_elites"] = len(qd_archive.cells)
        res["repair_pred_mae"] = float(np.mean(controller.prediction_errors)) if controller.prediction_errors else 0.0
        return res

    # Fusion (hybrid_exact or gated_specialist_routing)
    fusion_strategy = "gated_specialist_routing" if condition == "gated_fusion_ewc" else "hybrid_exact"
    fused, prov, parent_maps = fuse_parents(lineages, last_diags, strategy=fusion_strategy, seed=seed + 500)
    pre_adapt = evaluate_brain(fused, val_seeds[:2], stage="pre_adaptation", max_steps=max_episode_steps)

    if adapt_budget > 0:
        if trainer == "bptt":
            used += _train_brain(fused, adapt_budget, update_rec=False, struct_pl=False)
        else:
            used += run_adaptation_stage(fused, train_seeds, steps=adapt_budget, max_episode_steps=max_episode_steps)
    post_adapt = evaluate_brain(fused, val_seeds[:2], stage="post_adaptation", max_steps=max_episode_steps)

    replay_info: Dict[str, Any] = {}
    if replay_budget > 0:
        replay_mode = {
            "fusion_ordinary_replay": "sensory_action",
            "fusion_activity_replay": "sensory_activity",
            "full_dtr_finetune": "full_dtr",
            "rewired_control": "full_dtr",
            "ewc_dtr_finetune": "ewc_staged",
            "gated_fusion_ewc": "ewc_staged",
        }[condition]
        if trainer == "bptt":
            replay_info = run_bptt_dtr(
                fused, last_diags, parent_maps, mode=replay_mode,
                steps=replay_budget, max_episode_steps=max_episode_steps,
                prioritize=is_advanced, estimate_fisher=is_advanced,
            )
        else:
            replay_info = run_developmental_trace_replay(
                fused,
                last_diags,
                parent_maps,
                mode="full_dtr" if replay_mode == "ewc_staged" else replay_mode,
                steps=replay_budget,
                max_episode_steps=max_episode_steps,
            )
        used += replay_info.get("replay_steps", 0)

    if finetune_budget > 0:
        if trainer == "bptt" and is_advanced:
            used += train_ewc_finetune_steps(
                fused, train_seeds, steps=finetune_budget,
                fisher_diag=replay_info.get("fisher_diag"),
                ewc_lambda=30.0, rec_lr_scale=0.10, max_episode_steps=max_episode_steps,
            )
        elif trainer == "bptt":
            used += _train_brain(fused, finetune_budget, update_rec=True, struct_pl=False)
        else:
            used += run_autonomous_finetune(fused, train_seeds, steps=finetune_budget, max_episode_steps=max_episode_steps)

    sealed_metrics = evaluate_brain(fused, sealed_seeds, stage="sealed", max_steps=max_episode_steps)
    res = _format_result(condition, seed, used, t0, fused, sealed_metrics)
    res.update({
        "exact_inheritance_verified": prov.exact_inheritance_verified,
        "inherited_weight_hash": prov.inherited_weight_hash,
        "novel_matches": prov.novel_matches,
        "unmatched_novel_kept": prov.unmatched_novel_kept,
        "conflicts_recorded": prov.conflicts_recorded,
        "pre_adapt_reward": pre_adapt.mean_reward,
        "post_adapt_reward": post_adapt.mean_reward,
        "replay_activity_similarity": replay_info.get("activity_similarity", 0.0),
        "dag_nodes": len(dag),
        "qd_elites": len(qd_archive.cells),
        "repair_pred_mae": float(np.mean(controller.prediction_errors)) if controller.prediction_errors else 0.0,
    })
    return res


def _format_result(
    condition: str,
    seed: int,
    used: int,
    t0: float,
    brain: ConnectomeBrain,
    metrics: StageMetrics,
) -> Dict[str, Any]:
    return {
        "condition": condition,
        "seed": seed,
        "env_steps_used": used,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "energy_used": round(brain.energy_used, 4),
        "n_active_nodes": brain.n_active,
        "n_novel_edges": len(brain.overlay.novel_edges),
        "n_pruned_edges": len(brain.overlay.pruned_bio_edges),
        "sealed_mean_reward": metrics.mean_reward,
        "sealed_success_rate": metrics.success_rate,
        "sealed_pred_err": metrics.mean_pred_err,
        "per_task_reward": metrics.per_task_reward,
        "per_task_success": metrics.per_task_success,
        "exact_inheritance_verified": False,
    }


def run_successive_action_experiment(
    ref: ConnectomeReference,
    seeds: List[int],
    max_episode_steps: int = 80,
) -> Dict[str, Any]:
    """Compare successive-action descendant handoff (explicit state/pred/memory) vs 1 recurrent brain."""
    budgets = SubstrateBudgets()
    base_w = ref.to_initial_weights(budgets.row_norm_cap)
    single = ConnectomeBrain(ref, lineage_id="SINGLE_REC", generation=0, seed=seeds[0], budgets=budgets, shared_base_weights=base_w)
    descendants = [
        ConnectomeBrain(ref, lineage_id=f"DESC_{k}", generation=1, seed=seeds[0] + k + 1, budgets=budgets, shared_base_weights=base_w)
        for k in range(3)
    ]

    # Train both with equal steps
    train_brain_steps(single, seeds[:2], steps=600, max_episode_steps=max_episode_steps)
    for k, d in enumerate(descendants):
        train_brain_steps(d, seeds[:2], steps=200, max_episode_steps=max_episode_steps)

    env = ECCNRWorld(max_steps=max_episode_steps, seed=seeds[0])
    single_rews, handoff_rews = [], []
    for task in TASKS:
        for s in seeds[2:]:
            # Single recurrent
            obs = env.reset(task=task, seed=s)
            single.reset_state()
            tot_s = 0.0
            for _ in range(max_episode_steps):
                act, _ = single.forward(obs, explore_noise=0.0)
                res = env.step(act)
                tot_s += res.reward
                obs = res.obs
                if res.done:
                    break
            single_rews.append(tot_s)

            # Successive-action handoff: switch descendant every 27 steps, handing off rates & neuromod
            obs = env.reset(task=task, seed=s)
            for d in descendants:
                d.reset_state()
            tot_h = 0.0
            prev_rates = np.zeros(ref.n_nodes, dtype=np.float32)
            prev_mod = np.zeros(4, dtype=np.float32)
            active_k = 0
            for t in range(max_episode_steps):
                k = min(2, t // 27)
                if k != active_k:
                    # Explicit state handoff
                    n_c = min(len(prev_rates), descendants[k].n_active)
                    descendants[k].rates[:n_c] = prev_rates[:n_c]
                    descendants[k].neuromod[:] = prev_mod[:]
                    active_k = k
                act, _ = descendants[active_k].forward(obs, explore_noise=0.0)
                prev_rates = descendants[active_k].rates[: ref.n_nodes].copy()
                prev_mod = descendants[active_k].neuromod.copy()
                res = env.step(act)
                tot_h += res.reward
                obs = res.obs
                if res.done:
                    break
            handoff_rews.append(tot_h)

    return {
        "single_recurrent_mean_reward": float(np.mean(single_rews)),
        "successive_handoff_mean_reward": float(np.mean(handoff_rews)),
        "delta": float(np.mean(handoff_rews) - np.mean(single_rews)),
    }
