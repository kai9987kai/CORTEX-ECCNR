"""PyTorch CPU truncated-BPTT Advantage Actor-Critic (A2C) trainer for CORTEX-ECCNR.

Preserves exact FlyWire sparse topology (via binary adjacency mask), Dale's law
sign constraints, CoW weight semantics, and structural overlays, syncing trained
parameters back into ConnectomeBrain / RecurrentANNAnalogue.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ACTION_DIM, ECCNRWorld, OBS_DIM, TASKS
from .evolution import EpisodeTrace, ParentDiagnostics
from .substrate import ConnectomeBrain, RecurrentANNAnalogue


class _TorchConnectomeActorCritic(nn.Module):
    def __init__(self, brain: ConnectomeBrain, update_recurrent: bool = True):
        super().__init__()
        n_act = brain.n_active
        self.n_act = n_act
        self.n_bio = brain.ref.n_nodes
        self.update_recurrent = update_recurrent

        # Build dense masked W_rec preserving exact sparse FlyWire + overlay topology
        csr = brain._build_csr()
        w_dense = torch.from_numpy(csr.toarray().astype(np.float32))
        mask = (w_dense != 0.0).to(torch.float32)
        self.register_buffer("edge_mask", mask)

        # Pre-synaptic Dale's law signs per column
        signs = np.array([brain.node_sign(i) for i in range(n_act)], dtype=np.float32)
        self.register_buffer("pre_signs", torch.from_numpy(signs).unsqueeze(0))  # (1, n_act)

        self.w_rec = nn.Parameter(w_dense, requires_grad=update_recurrent)
        self.w_in = nn.Parameter(torch.from_numpy(brain.w_in[:n_act].copy()))
        self.w_out = nn.Parameter(torch.from_numpy(brain.w_out[:, :n_act].copy()))
        self.w_pred = nn.Parameter(torch.from_numpy(brain.w_pred[:, :n_act].copy()))
        self.thresholds = nn.Parameter(torch.from_numpy(brain.thresholds[:n_act].copy()))
        self.v_head = nn.Linear(n_act, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.9))

        self.has_router = bool(brain.specialist_w_outs and brain.router_w is not None)
        if self.has_router:
            spec_stack = np.zeros((len(brain.specialist_w_outs), ACTION_DIM, n_act), dtype=np.float32)
            for k_i, w_sp in enumerate(brain.specialist_w_outs):
                cols = min(n_act, w_sp.shape[1])
                spec_stack[k_i, :, :cols] = w_sp[:, :cols]
            self.register_buffer("spec_w_outs", torch.from_numpy(spec_stack))
            self.router_w = nn.Parameter(torch.from_numpy(brain.router_w.copy()))
        else:
            self.router_w = None

    def step_cell(
        self,
        r: torch.Tensor,
        obs: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_eff = self.w_rec * self.edge_mask
        total = F.linear(r, w_eff) + F.linear(obs, self.w_in) - self.thresholds
        if guidance is not None:
            total = total + torch.clamp(guidance, -0.45, 0.45)
        new_r = torch.tanh(0.6 * r + 0.4 * total)
        if self.has_router and self.router_w is not None:
            # 6 task-onehot + 4 zero-padded neuromod proxy from obs tail/proprioception
            ctx = torch.cat([obs[143:149], obs[135:139]], dim=0)
            alphas = F.softmax(F.linear(ctx, self.router_w), dim=0)  # (K,)
            spec_acts = torch.einsum("kan,n->ka", self.spec_w_outs, new_r)  # (K, 5)
            routed = torch.einsum("k,ka->a", alphas, spec_acts)
            mean_act = torch.tanh(0.5 * F.linear(new_r, self.w_out) + 0.5 * routed)
        else:
            mean_act = torch.tanh(F.linear(new_r, self.w_out))
        pred = F.linear(new_r, self.w_pred)
        val = self.v_head(new_r).squeeze(-1)
        return new_r, mean_act, pred, val

    @torch.no_grad()
    def enforce_dales_law(self) -> None:
        if not self.update_recurrent:
            return
        self.w_rec.mul_(self.edge_mask)
        nonzero = self.pre_signs != 0
        # Enforce sign per presynaptic column
        pos_mask = (self.pre_signs > 0).expand_as(self.w_rec)
        neg_mask = (self.pre_signs < 0).expand_as(self.w_rec)
        zero_mask = (~nonzero).expand_as(self.w_rec)
        self.w_rec.data[pos_mask] = torch.clamp(self.w_rec.data[pos_mask], 0.0, 1.5)
        self.w_rec.data[neg_mask] = torch.clamp(self.w_rec.data[neg_mask], -1.5, 0.0)
        self.w_rec.data[zero_mask] = 0.0

    @torch.no_grad()
    def sync_to_brain(self, brain: ConnectomeBrain) -> None:
        n_act = self.n_act
        if self.update_recurrent:
            w_np = (self.w_rec.detach() * self.edge_mask).cpu().numpy().astype(np.float32)
            bio_w = brain._ensure_cow_weights()
            post_idx = np.repeat(np.arange(brain.ref.n_nodes, dtype=np.int32), np.diff(brain.ref.indptr))
            bio_w[:] = w_np[post_idx, brain.ref.indices]
            for (post, pre), (old_w, inn_id, comp) in list(brain.overlay.novel_edges.items()):
                if post < n_act and pre < n_act:
                    brain.overlay.novel_edges[(post, pre)] = (float(w_np[post, pre]), inn_id, comp)
        brain.w_in[:n_act] = np.clip(self.w_in.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        brain.w_out[:, :n_act] = np.clip(self.w_out.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        brain.w_pred[:, :n_act] = np.clip(self.w_pred.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        brain.thresholds[:n_act] = np.clip(self.thresholds.detach().cpu().numpy(), -0.5, 0.5).astype(np.float32)
        if self.has_router and self.router_w is not None:
            brain.router_w = np.clip(self.router_w.detach().cpu().numpy(), -4.0, 4.0).astype(np.float32)
        brain._invalidate_csr()


class _TorchANNActorCritic(nn.Module):
    def __init__(self, ann: RecurrentANNAnalogue):
        super().__init__()
        h = ann.hidden_dim
        self.w_rec = nn.Parameter(torch.from_numpy(ann.w_rec.copy()))
        self.w_in = nn.Parameter(torch.from_numpy(ann.w_in.copy()))
        self.w_out = nn.Parameter(torch.from_numpy(ann.w_out.copy()))
        self.w_pred = nn.Parameter(torch.from_numpy(ann.w_pred.copy()))
        self.v_head = nn.Linear(h, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.9))

    def step_cell(self, r: torch.Tensor, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_r = torch.tanh(0.6 * r + 0.4 * (F.linear(r, self.w_rec) + F.linear(obs, self.w_in)))
        mean_act = torch.tanh(F.linear(new_r, self.w_out))
        pred = F.linear(new_r, self.w_pred)
        val = self.v_head(new_r).squeeze(-1)
        return new_r, mean_act, pred, val

    @torch.no_grad()
    def sync_to_ann(self, ann: RecurrentANNAnalogue) -> None:
        ann.w_rec[:] = np.clip(self.w_rec.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        ann.w_in[:] = np.clip(self.w_in.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        ann.w_out[:] = np.clip(self.w_out.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)
        ann.w_pred[:] = np.clip(self.w_pred.detach().cpu().numpy(), -2.0, 2.0).astype(np.float32)


def train_bptt_steps(
    brain: ConnectomeBrain,
    train_seeds: List[int],
    steps: int,
    max_episode_steps: int = 60,
    lr: float = 3e-3,
    update_recurrent: bool = True,
    structural_plasticity: bool = False,
    bptt_window: int = 20,
    gamma: float = 0.96,
) -> int:
    """Train ConnectomeBrain with truncated-BPTT A2C for an exact env-step budget."""
    model = _TorchConnectomeActorCritic(brain, update_recurrent=update_recurrent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])

    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        obs_np = env.reset(task=task, seed=seed)
        r = torch.zeros(model.n_act, dtype=torch.float32)

        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        pred_losses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        done = False
        for t in range(max_episode_steps):
            obs_t = torch.from_numpy(obs_np)
            r, mean_act, pred, val = model.step_cell(r, obs_t)
            std = torch.exp(torch.clamp(model.log_std, -2.5, 0.0))
            dist = torch.distributions.Normal(mean_act, std)
            act_sample = dist.sample()
            log_p = dist.log_prob(act_sample).sum()
            ent = dist.entropy().sum()

            act_np = np.clip(act_sample.detach().numpy(), -1.0, 1.0).astype(np.float32)
            res = env.step(act_np)
            used += 1

            target_np = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
            pred_losses.append(F.mse_loss(pred, torch.from_numpy(target_np)))
            log_probs.append(log_p)
            values.append(val)
            rewards.append(res.reward)
            entropies.append(ent)
            obs_np = res.obs

            brain.energy_used += float(torch.sum(torch.abs(r)).item()) * 1e-4 + float(model.edge_mask.sum().item()) * 1e-6

            if len(rewards) >= bptt_window or res.done or used >= steps:
                R = 0.0 if res.done else float(val.detach().item())
                returns: List[float] = []
                for rew in reversed(rewards):
                    R = rew + gamma * R
                    returns.insert(0, R)
                ret_t = torch.tensor(returns, dtype=torch.float32)
                val_t = torch.stack(values)
                adv = ret_t - val_t.detach()
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                pi_loss = -(torch.stack(log_probs) * adv).mean()
                v_loss = F.mse_loss(val_t, ret_t)
                p_loss = torch.stack(pred_losses).mean()
                ent_loss = -0.01 * torch.stack(entropies).mean()
                loss = pi_loss + 0.5 * v_loss + 0.25 * p_loss + ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.enforce_dales_law()

                r = r.detach()
                log_probs.clear()
                values.clear()
                rewards.clear()
                pred_losses.clear()
                entropies.clear()

            if res.done or used >= steps:
                done = True
                break

        if structural_plasticity and (ep_idx % 8 == 7):
            r_np = r.detach().numpy()
            top_act = np.argsort(np.abs(r_np))[::-1]
            if len(top_act) >= 2:
                brain.add_edge(int(top_act[0]), int(top_act[1]), birth_event="bptt_hebbian_synaptogenesis")
        ep_idx += 1

    model.sync_to_brain(brain)
    return used


def train_ann_bptt_steps(
    ann: RecurrentANNAnalogue,
    train_seeds: List[int],
    steps: int,
    max_episode_steps: int = 60,
    lr: float = 3e-3,
    bptt_window: int = 20,
    gamma: float = 0.96,
) -> int:
    """Train RecurrentANNAnalogue with truncated-BPTT A2C for an exact env-step budget."""
    model = _TorchANNActorCritic(ann)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])

    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        obs_np = env.reset(task=task, seed=seed)
        r = torch.zeros(ann.hidden_dim, dtype=torch.float32)

        log_probs, values, rewards, pred_losses, entropies = [], [], [], [], []
        for _ in range(max_episode_steps):
            obs_t = torch.from_numpy(obs_np)
            r, mean_act, pred, val = model.step_cell(r, obs_t)
            std = torch.exp(torch.clamp(model.log_std, -2.5, 0.0))
            dist = torch.distributions.Normal(mean_act, std)
            act_sample = dist.sample()

            act_np = np.clip(act_sample.detach().numpy(), -1.0, 1.0).astype(np.float32)
            res = env.step(act_np)
            used += 1

            target_np = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
            pred_losses.append(F.mse_loss(pred, torch.from_numpy(target_np)))
            log_probs.append(dist.log_prob(act_sample).sum())
            values.append(val)
            rewards.append(res.reward)
            entropies.append(dist.entropy().sum())
            obs_np = res.obs

            ann.energy_used += float(torch.sum(torch.abs(r)).item()) * 1e-4 + float(ann.hidden_dim ** 2) * 1e-6

            if len(rewards) >= bptt_window or res.done or used >= steps:
                R = 0.0 if res.done else float(val.detach().item())
                returns: List[float] = []
                for rew in reversed(rewards):
                    R = rew + gamma * R
                    returns.insert(0, R)
                ret_t = torch.tensor(returns, dtype=torch.float32)
                val_t = torch.stack(values)
                adv = ret_t - val_t.detach()
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                loss = (
                    -(torch.stack(log_probs) * adv).mean()
                    + 0.5 * F.mse_loss(val_t, ret_t)
                    + 0.25 * torch.stack(pred_losses).mean()
                    - 0.01 * torch.stack(entropies).mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                r = r.detach()
                log_probs.clear(); values.clear(); rewards.clear(); pred_losses.clear(); entropies.clear()

            if res.done or used >= steps:
                break
        ep_idx += 1

    model.sync_to_ann(ann)
    return used


def run_bptt_dtr(
    brain: ConnectomeBrain,
    parent_diagnostics: List[ParentDiagnostics],
    parent_maps: Dict[int, Dict[int, float]],
    mode: str = "full_dtr",
    steps: int = 2400,
    guidance_strength: float = 0.35,
    max_episode_steps: int = 60,
    lr: float = 3e-3,
    bptt_window: int = 20,
    gamma: float = 0.96,
    prioritize: bool = False,
    estimate_fisher: bool = False,
) -> Dict[str, Any]:
    """DTR with PyTorch truncated-BPTT consolidation, optional prioritized trace ranking, and Fisher estimation."""
    selected_traces: List[tuple[int, EpisodeTrace]] = []
    for p_idx, diag in enumerate(parent_diagnostics):
        wins = [t for t in diag.traces if t.outcome_label == "win"][:3]
        innov = [t for t in diag.traces if t.outcome_label == "innovative"][:2]
        fails = [t for t in diag.traces if t.outcome_label == "failure"][:2]
        for tr in wins + innov + fails:
            if tr.keyframes:
                selected_traces.append((p_idx, tr))

    if prioritize and selected_traces:
        selected_traces.sort(
            key=lambda pair: (1.5 if pair[1].success else 0.0) + pair[1].total_reward + 0.35 * pair[1].novelty_score,
            reverse=True,
        )

    if not selected_traces:
        return {"replay_steps": 0, "activity_similarity": 0.0, "fisher_diag": None}

    update_rec = (mode == "full_dtr")
    # If estimate_fisher=True on a frozen-core mode, enable grad on w_rec to measure g^2 without updating w_rec!
    model = _TorchConnectomeActorCritic(brain, update_recurrent=(update_rec or estimate_fisher))
    if estimate_fisher and not update_rec:
        trainable_params = [p for n, p in model.named_parameters() if n != "w_rec"]
    else:
        trainable_params = list(model.parameters())
    opt = torch.optim.Adam(trainable_params, lr=lr)
    fisher_accum = torch.zeros_like(model.w_rec.data) if estimate_fisher else None
    fisher_batches = 0
    env = ECCNRWorld(max_steps=max_episode_steps, seed=42)

    used = 0
    ep_idx = 0
    act_sims: List[float] = []

    while used < steps:
        p_idx, tr = selected_traces[ep_idx % len(selected_traces)]
        progress = min(1.0, used / max(1, steps))
        anneal = max(0.0, 1.0 - progress)
        node_map = parent_maps.get(p_idx, {})
        kf_by_t = {kf.t: kf for kf in tr.keyframes}

        obs_np = env.reset(task=tr.task, seed=tr.seed)
        r = torch.zeros(model.n_act, dtype=torch.float32)
        log_probs, values, rewards, pred_losses, imitate_losses = [], [], [], [], []

        for t in range(max_episode_steps):
            guidance_t = None
            kf = kf_by_t.get(t)
            if kf is not None and mode in ("sensory_activity", "full_dtr", "ewc_staged") and anneal > 0.0:
                g = np.zeros(model.n_act, dtype=np.float32)
                r_curr = r.detach().numpy()
                for src_idx, dst_idx in node_map.items():
                    s_i, d_i = int(src_idx), int(dst_idx)
                    if s_i < len(kf.rates) and d_i < model.n_act:
                        g[d_i] = float(kf.rates[s_i] - r_curr[d_i])
                scale = (0.4 * guidance_strength) if mode == "ewc_staged" else guidance_strength
                guidance_t = torch.from_numpy((scale * anneal * g).astype(np.float32))

            r, mean_act, pred, val = model.step_cell(r, torch.from_numpy(obs_np), guidance=guidance_t)

            if kf is not None:
                n_cmp = min(brain.ref.n_nodes, len(kf.rates))
                r_f = r.detach().numpy()[:n_cmp]
                r_p = kf.rates[:n_cmp]
                denom = float(np.linalg.norm(r_f) * np.linalg.norm(r_p))
                if denom > 1e-6:
                    act_sims.append(float(np.dot(r_f, r_p) / denom))
                if mode in ("sensory_action", "full_dtr", "ewc_staged") and anneal > 0.0:
                    imitate_losses.append(anneal * F.mse_loss(mean_act, torch.from_numpy(kf.action)))

            std = torch.exp(torch.clamp(model.log_std, -2.5, 0.0))
            dist = torch.distributions.Normal(mean_act, std)
            act_sample = dist.sample()
            act_np = np.clip(act_sample.detach().numpy(), -1.0, 1.0).astype(np.float32)
            res = env.step(act_np)
            used += 1

            target_np = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
            pred_losses.append(F.mse_loss(pred, torch.from_numpy(target_np)))
            log_probs.append(dist.log_prob(act_sample).sum())
            values.append(val)
            rewards.append(res.reward)
            obs_np = res.obs

            if len(rewards) >= bptt_window or res.done or used >= steps:
                R = 0.0 if res.done else float(val.detach().item())
                returns: List[float] = []
                for rew in reversed(rewards):
                    R = rew + gamma * R
                    returns.insert(0, R)
                ret_t = torch.tensor(returns, dtype=torch.float32)
                val_t = torch.stack(values)
                adv = ret_t - val_t.detach()
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                loss = (
                    -(torch.stack(log_probs) * adv).mean()
                    + 0.5 * F.mse_loss(val_t, ret_t)
                    + 0.25 * torch.stack(pred_losses).mean()
                )
                if imitate_losses:
                    loss = loss + 0.35 * torch.stack(imitate_losses).mean()
                opt.zero_grad()
                if model.w_rec.grad is not None:
                    model.w_rec.grad.zero_()
                loss.backward()
                if fisher_accum is not None and model.w_rec.grad is not None:
                    fisher_accum.add_((model.w_rec.grad.detach() * model.edge_mask) ** 2)
                    fisher_batches += 1
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                if update_rec:
                    model.enforce_dales_law()
                r = r.detach()
                log_probs.clear(); values.clear(); rewards.clear(); pred_losses.clear(); imitate_losses.clear()

            if res.done or used >= steps:
                break
        ep_idx += 1

    model.update_recurrent = update_rec
    model.sync_to_brain(brain)
    fisher_out = None
    if fisher_accum is not None and fisher_batches > 0:
        f_mean = (fisher_accum / fisher_batches).cpu().numpy()
        f_max = float(np.max(f_mean))
        fisher_out = (f_mean / max(1e-8, f_max)).astype(np.float32) if f_max > 0 else f_mean.astype(np.float32)
    return {
        "replay_steps": used,
        "activity_similarity": float(np.mean(act_sims)) if act_sims else 0.0,
        "fisher_diag": fisher_out,
    }


def train_ewc_finetune_steps(
    brain: ConnectomeBrain,
    train_seeds: List[int],
    steps: int,
    fisher_diag: Optional[np.ndarray] = None,
    ewc_lambda: float = 25.0,
    rec_lr_scale: float = 0.10,
    max_episode_steps: int = 60,
    lr: float = 3e-3,
    bptt_window: int = 20,
    gamma: float = 0.96,
) -> int:
    """Two-stage protected autonomous fine-tuning: 0.1x differential recurrent LR + EWC Fisher anchor."""
    model = _TorchConnectomeActorCritic(brain, update_recurrent=True)
    w_rec_anchor = model.w_rec.detach().clone()
    if fisher_diag is not None and fisher_diag.shape == tuple(w_rec_anchor.shape):
        fisher_t = torch.from_numpy(fisher_diag.astype(np.float32))
    else:
        fisher_t = model.edge_mask.clone()

    fast_params = [p for n, p in model.named_parameters() if n != "w_rec"]
    opt = torch.optim.Adam([
        {"params": fast_params, "lr": lr},
        {"params": [model.w_rec], "lr": lr * rec_lr_scale},
    ])
    env = ECCNRWorld(max_steps=max_episode_steps, seed=train_seeds[0])

    used = 0
    ep_idx = 0
    while used < steps:
        task = TASKS[ep_idx % len(TASKS)]
        seed = train_seeds[(ep_idx // len(TASKS)) % len(train_seeds)]
        obs_np = env.reset(task=task, seed=seed)
        r = torch.zeros(model.n_act, dtype=torch.float32)
        log_probs, values, rewards, pred_losses, entropies = [], [], [], [], []

        for _ in range(max_episode_steps):
            r, mean_act, pred, val = model.step_cell(r, torch.from_numpy(obs_np))
            std = torch.exp(torch.clamp(model.log_std, -2.5, 0.0))
            dist = torch.distributions.Normal(mean_act, std)
            act_sample = dist.sample()
            act_np = np.clip(act_sample.detach().numpy(), -1.0, 1.0).astype(np.float32)
            res = env.step(act_np)
            used += 1

            target_np = np.concatenate([res.obs[135:143], [res.reward]]).astype(np.float32)
            pred_losses.append(F.mse_loss(pred, torch.from_numpy(target_np)))
            log_probs.append(dist.log_prob(act_sample).sum())
            values.append(val)
            rewards.append(res.reward)
            entropies.append(dist.entropy().sum())
            obs_np = res.obs

            brain.energy_used += float(torch.sum(torch.abs(r)).item()) * 1e-4 + float(model.edge_mask.sum().item()) * 1e-6

            if len(rewards) >= bptt_window or res.done or used >= steps:
                R = 0.0 if res.done else float(val.detach().item())
                returns: List[float] = []
                for rew in reversed(rewards):
                    R = rew + gamma * R
                    returns.insert(0, R)
                ret_t = torch.tensor(returns, dtype=torch.float32)
                val_t = torch.stack(values)
                adv = ret_t - val_t.detach()
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                ewc_loss = 0.5 * ewc_lambda * torch.mean(fisher_t * ((model.w_rec - w_rec_anchor) ** 2))
                loss = (
                    -(torch.stack(log_probs) * adv).mean()
                    + 0.5 * F.mse_loss(val_t, ret_t)
                    + 0.25 * torch.stack(pred_losses).mean()
                    - 0.01 * torch.stack(entropies).mean()
                    + ewc_loss
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.enforce_dales_law()
                r = r.detach()
                log_probs.clear(); values.clear(); rewards.clear(); pred_losses.clear(); entropies.clear()

            if res.done or used >= steps:
                break
        ep_idx += 1

    model.sync_to_brain(brain)
    return used
