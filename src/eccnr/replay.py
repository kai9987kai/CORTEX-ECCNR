"""Pre-adaptation, Developmental Trace Replay (DTR), and sealed evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .env import ECCNRWorld, TASKS
from .evolution import EpisodeTrace, ParentDiagnostics, run_episode_and_capture
from .substrate import ConnectomeBrain


@dataclass
class StageMetrics:
    stage: str
    mean_reward: float
    success_rate: float
    mean_pred_err: float
    per_task_reward: Dict[str, float]
    per_task_success: Dict[str, float]
    energy_used: float
    env_steps: int


def evaluate_brain(
    brain: ConnectomeBrain,
    seeds: List[int],
    stage: str = "eval",
    max_steps: int = 80,
) -> StageMetrics:
    """Frozen evaluation on sealed or validation seeds."""
    env = ECCNRWorld(max_steps=max_steps, seed=seeds[0])
    per_task_rew: Dict[str, List[float]] = {t: [] for t in TASKS}
    per_task_succ: Dict[str, List[float]] = {t: [] for t in TASKS}
    pred_errs: List[float] = []
    e_before = brain.energy_used

    for task in TASKS:
        for s in seeds:
            tr = run_episode_and_capture(
                brain=brain,
                env=env,
                task=task,
                seed=s,
                train=False,
                explore_noise=0.0,
                keyframe_stride=max_steps + 1,
            )
            per_task_rew[task].append(tr.total_reward)
            per_task_succ[task].append(1.0 if tr.success else 0.0)
            pred_errs.append(tr.mean_pred_err)

    task_rew = {t: float(np.mean(v)) for t, v in per_task_rew.items()}
    task_succ = {t: float(np.mean(v)) for t, v in per_task_succ.items()}
    return StageMetrics(
        stage=stage,
        mean_reward=float(np.mean(list(task_rew.values()))),
        success_rate=float(np.mean(list(task_succ.values()))),
        mean_pred_err=float(np.mean(pred_errs)) if pred_errs else 0.0,
        per_task_reward=task_rew,
        per_task_success=task_succ,
        energy_used=float(brain.energy_used - e_before),
        env_steps=len(TASKS) * len(seeds) * max_steps,
    )


def run_adaptation_stage(
    brain: ConnectomeBrain,
    train_seeds: List[int],
    steps: int = 400,
    max_episode_steps: int = 80,
    lr: float = 0.025,
) -> int:
    """Freeze inherited recurrent weights; train only readout/prediction/routing adapters."""
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])
    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        run_episode_and_capture(
            brain=brain,
            env=env,
            task=task,
            seed=seed,
            train=True,
            lr=lr,
            update_recurrent=False,  # Freeze inherited recurrent weights!
            explore_noise=0.04,
            keyframe_stride=max_episode_steps + 1,
        )
        used += max_episode_steps
        ep_idx += 1
    return used


def run_developmental_trace_replay(
    brain: ConnectomeBrain,
    parent_diagnostics: List[ParentDiagnostics],
    parent_maps: Dict[int, Dict[int, float]],
    mode: str = "full_dtr",
    steps: int = 400,
    guidance_strength: float = 0.35,
    max_episode_steps: int = 80,
    lr: float = 0.02,
) -> Dict[str, Any]:
    """Replay parent wins & useful failures with mapped soft current guidance annealed to zero."""
    selected_traces: List[tuple[int, EpisodeTrace]] = []
    for p_idx, diag in enumerate(parent_diagnostics):
        wins = [t for t in diag.traces if t.outcome_label == "win"][:3]
        innov = [t for t in diag.traces if t.outcome_label == "innovative"][:2]
        fails = [t for t in diag.traces if t.outcome_label == "failure"][:2]
        for tr in wins + innov + fails:
            if tr.keyframes:
                selected_traces.append((p_idx, tr))

    if not selected_traces:
        return {"replay_steps": 0, "activity_similarity": 0.0}

    env = ECCNRWorld(max_steps=max_episode_steps, seed=42)
    used = 0
    ep_idx = 0
    act_sims: List[float] = []

    while used < steps:
        p_idx, tr = selected_traces[ep_idx % len(selected_traces)]
        progress = min(1.0, used / max(1, steps))
        anneal = max(0.0, 1.0 - progress)
        node_map = parent_maps.get(p_idx, {})

        obs = env.reset(task=tr.task, seed=tr.seed)
        brain.reset_state()
        kf_by_t = {kf.t: kf for kf in tr.keyframes}

        for t in range(max_episode_steps):
            guidance = None
            kf = kf_by_t.get(t)
            if kf is not None and mode in ("sensory_activity", "full_dtr") and anneal > 0.0:
                g = np.zeros(brain.n_active, dtype=np.float32)
                for src_idx, dst_idx in node_map.items():
                    s_i, d_i = int(src_idx), int(dst_idx)
                    if s_i < len(kf.rates) and d_i < brain.n_active:
                        g[d_i] = float(kf.rates[s_i] - brain.rates[d_i])
                guidance = np.clip(guidance_strength * anneal * g, -0.45, 0.45)

            action, pred = brain.forward(obs, external_guidance=guidance, explore_noise=0.02)

            if kf is not None:
                n_cmp = min(brain.ref.n_nodes, len(kf.rates))
                r_f = brain.rates[:n_cmp]
                r_p = kf.rates[:n_cmp]
                denom = float(np.linalg.norm(r_f) * np.linalg.norm(r_p))
                if denom > 1e-6:
                    act_sims.append(float(np.dot(r_f, r_p) / denom))

                if mode in ("sensory_action", "full_dtr") and anneal > 0.0:
                    action = np.clip((1.0 - 0.3 * anneal) * action + (0.3 * anneal) * kf.action, -1.0, 1.0).astype(np.float32)

            res = env.step(action)
            update_rec = (mode == "full_dtr")
            brain.update_plasticity(
                obs=obs,
                action_taken=action,
                pred=pred,
                next_obs=res.obs,
                reward=res.reward,
                lr=lr,
                update_recurrent=update_rec,
            )
            used += 1
            obs = res.obs
            if res.done or used >= steps:
                break
        ep_idx += 1

    return {
        "replay_steps": used,
        "activity_similarity": float(np.mean(act_sims)) if act_sims else 0.0,
    }


def run_autonomous_finetune(
    brain: ConnectomeBrain,
    train_seeds: List[int],
    steps: int = 400,
    max_episode_steps: int = 80,
    lr: float = 0.015,
) -> int:
    """Remove all replay forcing and supervision; train with ordinary simulated rewards."""
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])
    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        run_episode_and_capture(
            brain=brain,
            env=env,
            task=task,
            seed=seed,
            train=True,
            lr=lr,
            update_recurrent=True,
            explore_noise=0.03,
            keyframe_stride=max_episode_steps + 1,
        )
        used += max_episode_steps
        ep_idx += 1
    return used
