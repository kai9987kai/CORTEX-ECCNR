"""Parent diagnosis, repair controllers, lineage DAG, and developmental trace capture."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .env import ECCNRWorld, TASKS
from .substrate import ConnectomeBrain


@dataclass
class TraceKeyframe:
    t: int
    obs: np.ndarray
    action: np.ndarray
    reward: float
    pred: np.ndarray
    rates: np.ndarray
    neuromod: np.ndarray


@dataclass
class EpisodeTrace:
    task: str
    seed: int
    total_reward: float
    success: bool
    mean_pred_err: float
    novelty_score: float
    outcome_label: str  # "win" | "failure" | "innovative"
    keyframes: List[TraceKeyframe] = field(default_factory=list)


@dataclass
class ParentDiagnostics:
    lineage_id: str
    generation: int
    mean_reward: float
    success_rate: float
    mean_pred_err: float
    regret: float
    per_task_reward: Dict[str, float]
    per_task_success: Dict[str, float]
    causal_node_sensitivity: np.ndarray  # shape (n_active,)
    mean_activation_profile: np.ndarray  # shape (n_active,)
    traces: List[EpisodeTrace] = field(default_factory=list)
    behavior_signature: np.ndarray = field(default_factory=lambda: np.zeros(len(TASKS), dtype=np.float32))


@dataclass
class RepairProposal:
    edit_kind: str  # "add_neuron" | "add_edge" | "prune" | "compartment" | "readout_nudge"
    target_task: str
    predicted_delta: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageNode:
    node_id: str
    lineage_id: str
    generation: int
    parent_ids: List[str]
    repair_kind: str
    predicted_fix_delta: float
    actual_fix_delta: float
    mean_reward: float
    success_rate: float
    weight_hash: str
    novelty_score: float
    snapshot: Optional[Dict[str, Any]] = None


def run_episode_and_capture(
    brain: ConnectomeBrain,
    env: ECCNRWorld,
    task: str,
    seed: int,
    train: bool = True,
    lr: float = 0.02,
    update_recurrent: bool = True,
    explore_noise: float = 0.05,
    keyframe_stride: int = 4,
    max_trace_steps: int = 64,
    external_guidance_fn: Optional[Any] = None,
) -> EpisodeTrace:
    """Run one episode, optionally updating online plasticity, and capture compressed trace keyframes."""
    obs = env.reset(task=task, seed=seed)
    brain.reset_state()
    total_reward = 0.0
    pred_errs: List[float] = []
    keyframes: List[TraceKeyframe] = []
    success = False
    act_norms: List[float] = []

    for step_idx in range(env.max_steps):
        guidance = external_guidance_fn(step_idx, brain) if external_guidance_fn is not None else None
        action, pred = brain.forward(obs, external_guidance=guidance, explore_noise=explore_noise if train else 0.0)
        res = env.step(action)
        total_reward += res.reward
        act_norms.append(float(np.linalg.norm(action)))

        if train:
            perr = brain.update_plasticity(
                obs=obs,
                action_taken=action,
                pred=pred,
                next_obs=res.obs,
                reward=res.reward,
                lr=lr,
                update_recurrent=update_recurrent,
            )
        else:
            target = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
            perr = float(np.mean((target - pred) ** 2))
        pred_errs.append(perr)

        if step_idx % keyframe_stride == 0 and len(keyframes) < (max_trace_steps // keyframe_stride + 1):
            keyframes.append(
                TraceKeyframe(
                    t=step_idx,
                    obs=obs.copy(),
                    action=action.copy(),
                    reward=float(res.reward),
                    pred=pred.copy(),
                    rates=brain.rates[: brain.n_active].copy(),
                    neuromod=brain.neuromod.copy(),
                )
            )

        obs = res.obs
        if res.done:
            success = bool(res.info.get("success", False))
            break

    mean_perr = float(np.mean(pred_errs)) if pred_errs else 0.0
    novelty = float(np.std(act_norms) + 0.5 * mean_perr)
    if success:
        label = "win"
    elif novelty > 0.35 or total_reward > 0.2:
        label = "innovative"
    else:
        label = "failure"

    return EpisodeTrace(
        task=task,
        seed=seed,
        total_reward=float(total_reward),
        success=success,
        mean_pred_err=mean_perr,
        novelty_score=novelty,
        outcome_label=label,
        keyframes=keyframes,
    )


def diagnose_parent(
    brain: ConnectomeBrain,
    seeds: List[int],
    max_steps: int = 80,
    keyframe_stride: int = 4,
) -> ParentDiagnostics:
    """Evaluate parent across tasks, run counterfactual state-cloned perturbation tests, and capture traces."""
    env = ECCNRWorld(max_steps=max_steps, seed=seeds[0])
    per_task_rew: Dict[str, List[float]] = {t: [] for t in TASKS}
    per_task_succ: Dict[str, List[float]] = {t: [] for t in TASKS}
    traces: List[EpisodeTrace] = []
    act_accum = np.zeros(brain.n_active, dtype=np.float64)
    act_count = 0

    for task in TASKS:
        for s in seeds:
            tr = run_episode_and_capture(
                brain=brain,
                env=env,
                task=task,
                seed=s,
                train=False,
                explore_noise=0.0,
                keyframe_stride=keyframe_stride,
            )
            traces.append(tr)
            per_task_rew[task].append(tr.total_reward)
            per_task_succ[task].append(1.0 if tr.success else 0.0)
            for kf in tr.keyframes:
                n_len = min(len(act_accum), len(kf.rates))
                act_accum[:n_len] += np.abs(kf.rates[:n_len])
                act_count += 1

    mean_act = (act_accum / max(1, act_count)).astype(np.float32)

    # Counterfactual perturbation credit assignment on a cloned state
    causal_sens = np.zeros(brain.n_active, dtype=np.float32)
    env.reset(task="navigation", seed=seeds[0])
    base_snap = env.clone_state()
    brain_snap = brain.snapshot()

    # Baseline 10-step rollout from cloned state
    def _short_rollout() -> float:
        brain.reset_state()
        tot = 0.0
        obs = env.observe()
        for _ in range(10):
            act, _ = brain.forward(obs, explore_noise=0.0)
            res = env.step(act)
            tot += res.reward
            obs = res.obs
            if res.done:
                break
        return tot

    base_ret = _short_rollout()
    # Probe top-16 most active nodes
    top_nodes = np.argsort(mean_act)[::-1][: min(16, brain.n_active)]
    for nid in top_nodes:
        env.restore_state(base_snap)
        brain.restore(brain_snap)
        brain.thresholds[int(nid)] += 1.0  # suppress node
        pert_ret = _short_rollout()
        causal_sens[int(nid)] = abs(base_ret - pert_ret)

    env.restore_state(base_snap)
    brain.restore(brain_snap)

    task_rew_means = {t: float(np.mean(v)) for t, v in per_task_rew.items()}
    task_succ_means = {t: float(np.mean(v)) for t, v in per_task_succ.items()}
    mean_reward = float(np.mean(list(task_rew_means.values())))
    success_rate = float(np.mean(list(task_succ_means.values())))
    mean_pred_err = float(np.mean([tr.mean_pred_err for tr in traces]))
    # Heuristic optimal return ~ 2.2 per episode
    regret = float(max(0.0, 2.2 - mean_reward))
    sig = np.array([task_rew_means[t] for t in TASKS], dtype=np.float32)

    return ParentDiagnostics(
        lineage_id=brain.lineage_id,
        generation=brain.generation,
        mean_reward=mean_reward,
        success_rate=success_rate,
        mean_pred_err=mean_pred_err,
        regret=regret,
        per_task_reward=task_rew_means,
        per_task_success=task_succ_means,
        causal_node_sensitivity=causal_sens,
        mean_activation_profile=mean_act,
        traces=traces,
        behavior_signature=sig,
    )


class LearnedRepairController:
    """Parent-conditioned repair controller that learns to predict whether an edit fixes a failure."""

    EDIT_KINDS = ("add_neuron", "add_edge", "prune", "compartment", "readout_nudge")

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Linear ridge model mapping [mean_reward, pred_err, regret, edit_onehot(5)] -> predicted delta_reward
        self.feat_dim = 3 + len(self.EDIT_KINDS)
        self.weights = np.zeros(self.feat_dim, dtype=np.float64)
        self.X_hist: List[np.ndarray] = []
        self.y_hist: List[float] = []
        self.prediction_errors: List[float] = []

    def _featurize(self, diag: ParentDiagnostics, edit_kind: str) -> np.ndarray:
        vec = np.zeros(self.feat_dim, dtype=np.float64)
        vec[0] = diag.mean_reward
        vec[1] = diag.mean_pred_err
        vec[2] = diag.regret
        vec[3 + self.EDIT_KINDS.index(edit_kind)] = 1.0
        return vec

    def propose(self, diag: ParentDiagnostics, mode: str = "learned") -> RepairProposal:
        worst_task = min(diag.per_task_reward, key=diag.per_task_reward.get)
        if mode == "random":
            kind = str(self.rng.choice(self.EDIT_KINDS))
            pred_delta = 0.0
        elif mode == "search":
            # Heuristic search: if pred_err high -> add_edge; if low reward -> add_neuron or readout_nudge
            if diag.mean_pred_err > 0.15:
                kind = "add_edge"
            elif diag.success_rate < 0.35:
                kind = "add_neuron"
            else:
                kind = "readout_nudge"
            pred_delta = 0.05
        elif mode == "ucb_learned":
            # Ridge posterior UCB: predicted delta + epistemic uncertainty bonus
            best_kind = "add_edge"
            best_score = -1e9
            if len(self.X_hist) >= 3:
                X = np.vstack(self.X_hist)
                cov = np.linalg.inv(X.T @ X + 0.1 * np.eye(self.feat_dim))
            else:
                cov = 2.0 * np.eye(self.feat_dim)
            for k in self.EDIT_KINDS:
                x = self._featurize(diag, k)
                mean_pred = float(self.weights @ x)
                unc = float(np.sqrt(max(1e-8, x @ cov @ x)))
                score = mean_pred + 0.25 * unc
                if score > best_score:
                    best_score = score
                    best_kind = k
            kind = best_kind
            pred_delta = float(self.weights @ self._featurize(diag, kind))
        else:
            # Learned controller scores each edit kind
            best_kind = "add_edge"
            best_score = -1e9
            for k in self.EDIT_KINDS:
                x = self._featurize(diag, k)
                score = float(self.weights @ x) + float(self.rng.normal(0.0, 0.02))
                if score > best_score:
                    best_score = score
                    best_kind = k
            kind = best_kind
            pred_delta = float(best_score)

        return RepairProposal(edit_kind=kind, target_task=worst_task, predicted_delta=pred_delta)

    def apply_repair(self, child: ConnectomeBrain, diag: ParentDiagnostics, proposal: RepairProposal) -> None:
        """Apply bounded developmental edit to child based on parent diagnosis."""
        top_sens = np.argsort(diag.causal_node_sensitivity)[::-1]
        top_act = np.argsort(diag.mean_activation_profile)[::-1]
        if proposal.edit_kind == "add_neuron":
            nid = child.add_neuron(
                role="central",
                sign=1 if child.rng.random() < 0.7 else -1,
                birth_event=f"repair_for_{proposal.target_task}",
                hypothesis=f"expand_circuit_for_{proposal.target_task}",
            )
            if nid is not None and len(top_act) >= 2:
                child.add_edge(nid, int(top_act[0]), birth_event="wire_new_neuron")
                child.add_edge(int(top_sens[0]) if len(top_sens) > 0 else 0, nid, birth_event="wire_new_neuron")
        elif proposal.edit_kind == "add_edge":
            for i in range(min(4, len(top_act))):
                post = int(top_sens[i % len(top_sens)])
                pre = int(top_act[(i + 1) % len(top_act)])
                child.add_edge(post, pre, birth_event=f"repair_edge_{proposal.target_task}")
        elif proposal.edit_kind == "prune":
            child.prune_weak_edges(threshold=0.015, max_prune=12)
        elif proposal.edit_kind == "compartment":
            for nid in top_sens[:4]:
                child.overlay.compartment_gains[int(nid)] = float(np.clip(1.0 + child.rng.normal(0.0, 0.15), 0.6, 1.5))
            child._invalidate_csr()
        elif proposal.edit_kind == "readout_nudge":
            n_act = child.n_active
            child.w_out[:, :n_act] += (child.rng.normal(0.0, 0.03, size=(5, n_act))).astype(np.float32)

    def observe_outcome(self, diag: ParentDiagnostics, proposal: RepairProposal, actual_delta: float) -> float:
        """Score controller's prediction after child evaluation and update ridge predictor."""
        x = self._featurize(diag, proposal.edit_kind)
        pred_err = abs(proposal.predicted_delta - actual_delta)
        self.prediction_errors.append(pred_err)
        self.X_hist.append(x)
        self.y_hist.append(actual_delta)
        if len(self.X_hist) >= 3:
            X = np.vstack(self.X_hist)
            y = np.array(self.y_hist, dtype=np.float64)
            reg = 0.1 * np.eye(self.feat_dim)
            self.weights = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return pred_err


@dataclass
class QDElite:
    niche_key: Tuple[str, int]  # (best_task, novelty_tier 0..2)
    lineage_id: str
    generation: int
    niche_score: float
    mean_reward: float
    novelty_score: float
    snapshot: Dict[str, Any]
    diagnostics: ParentDiagnostics


class QualityDiversityArchive:
    """MAP-Elites quality-diversity archive retaining specialist and high-novelty brains."""

    def __init__(self, novelty_bins: Tuple[float, float] = (0.20, 0.40)):
        self.novelty_bins = novelty_bins
        self.cells: Dict[Tuple[str, int], QDElite] = {}

    def _novelty_tier(self, nov: float) -> int:
        if nov < self.novelty_bins[0]:
            return 0
        if nov < self.novelty_bins[1]:
            return 1
        return 2

    def consider(self, brain: ConnectomeBrain, diag: ParentDiagnostics) -> bool:
        best_task = max(diag.per_task_reward, key=diag.per_task_reward.get)
        nov = float(np.mean([t.novelty_score for t in diag.traces])) if diag.traces else 0.0
        tier = self._novelty_tier(nov)
        key = (best_task, tier)
        score = float(diag.per_task_reward[best_task]) + 0.15 * nov
        existing = self.cells.get(key)
        if existing is None or score > existing.niche_score:
            self.cells[key] = QDElite(
                niche_key=key,
                lineage_id=brain.lineage_id,
                generation=brain.generation,
                niche_score=score,
                mean_reward=diag.mean_reward,
                novelty_score=nov,
                snapshot=brain.snapshot(),
                diagnostics=diag,
            )
            return True
        return False

    def summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "best_task": k[0],
                "novelty_tier": k[1],
                "lineage_id": e.lineage_id,
                "generation": e.generation,
                "niche_score": round(e.niche_score, 4),
                "mean_reward": round(e.mean_reward, 4),
                "novelty_score": round(e.novelty_score, 4),
            }
            for k, e in sorted(self.cells.items())
        ]


def cross_lineage_exchange(
    recipient: ConnectomeBrain,
    donor_snap: Dict[str, Any],
    donor_diag: ParentDiagnostics,
    max_graft_edges: int = 4,
) -> int:
    """Graft top causal-sensitivity biological/novel motifs from a specialist donor into recipient."""
    grafted = 0
    top_nodes = np.argsort(donor_diag.causal_node_sensitivity)[::-1][:4]
    donor_overlay = donor_snap["overlay"]
    for (post, pre), (w_val, _, comp) in list(donor_overlay.novel_edges.items()):
        if (post in top_nodes or pre in top_nodes) and post < recipient.n_active and pre < recipient.n_active:
            if (post, pre) not in recipient.overlay.novel_edges:
                recipient.add_edge(
                    post,
                    pre,
                    weight=w_val,
                    compartment=comp,
                    birth_event=f"cross_lineage_graft_from_{donor_snap['lineage_id']}",
                    hypothesis="specialist_motif_exchange",
                )
                grafted += 1
                if grafted >= max_graft_edges:
                    break
    return grafted
