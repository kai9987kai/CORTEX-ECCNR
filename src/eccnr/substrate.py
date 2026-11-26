"""Sparse FlyWire-derived rate connectome substrate with CoW weights and developmental dynamics.

Preserves the immutable biological reference graph while supporting:
- Copy-on-write (CoW) base edge weights
- Sparse structural overlays (neurogenesis, synaptogenesis, pruning, compartment gains)
- Persistent innovation IDs, birth events, function hypotheses, and mutation histories
- Dale's law sign enforcement, adaptive thresholds, bounded neuromodulation, and budgets
- Eligibility-trace reward-modulated plasticity + prediction head
- Exact checkpointing and rollback of weights, plasticity state, and structure
- Matched-capacity conventional recurrent ANN analogue for controlled comparison
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .connectome import ConnectomeReference
from .env import ACTION_DIM, OBS_DIM


@dataclass
class InnovationRecord:
    innovation_id: str
    kind: str  # "neuron" | "edge" | "prune" | "compartment"
    parent_lineage: str
    generation: int
    birth_event: str
    function_hypothesis: str
    mutation_history: List[str] = field(default_factory=list)


@dataclass
class StructuralOverlay:
    """Sparse developmental overlay on top of an immutable ConnectomeReference."""

    n_novel_neurons: int = 0
    novel_signs: List[int] = field(default_factory=list)
    novel_roles: List[str] = field(default_factory=list)
    novel_innovation_ids: List[str] = field(default_factory=list)
    novel_edges: Dict[Tuple[int, int], Tuple[float, str, int]] = field(default_factory=dict)
    # (post, pre) -> (weight, innovation_id, compartment_id)
    pruned_bio_edges: set[int] = field(default_factory=set)  # indices into ref.indices
    compartment_gains: Dict[int, float] = field(default_factory=dict)  # node_idx -> gain
    innovations: List[InnovationRecord] = field(default_factory=list)

    def copy(self) -> "StructuralOverlay":
        return StructuralOverlay(
            n_novel_neurons=self.n_novel_neurons,
            novel_signs=list(self.novel_signs),
            novel_roles=list(self.novel_roles),
            novel_innovation_ids=list(self.novel_innovation_ids),
            novel_edges=dict(self.novel_edges),
            pruned_bio_edges=set(self.pruned_bio_edges),
            compartment_gains=dict(self.compartment_gains),
            innovations=copy.deepcopy(self.innovations),
        )


@dataclass
class SubstrateBudgets:
    max_new_neurons: int = 32
    max_new_edges: int = 256
    firing_rate_cap: float = 1.0
    row_norm_cap: float = 0.90
    energy_budget: float = 1e6
    neuromod_min: float = -0.5
    neuromod_max: float = 0.5


class ConnectomeBrain:
    """Simulated FlyWire-derived rate connectome with CoW weights and developmental plasticity."""

    def __init__(
        self,
        ref: ConnectomeReference,
        lineage_id: str = "L0",
        generation: int = 0,
        seed: int = 42,
        budgets: Optional[SubstrateBudgets] = None,
        shared_base_weights: Optional[np.ndarray] = None,
        enable_delays: bool = False,
        enable_dendritic_gating: bool = False,
    ):
        self.ref = ref
        self.lineage_id = lineage_id
        self.generation = generation
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.budgets = budgets or SubstrateBudgets()
        self.enable_delays = bool(enable_delays)
        self.enable_dendritic_gating = bool(enable_dendritic_gating)

        # Copy-on-write biological weights
        if shared_base_weights is not None:
            self._bio_weights = shared_base_weights
            self._bio_weights_owned = False
        else:
            self._bio_weights = ref.to_initial_weights(row_norm_cap=self.budgets.row_norm_cap)
            self._bio_weights_owned = True

        self.overlay = StructuralOverlay()
        n = ref.n_nodes
        self.max_total_nodes = n + self.budgets.max_new_neurons

        # Pre-allocate node-level arrays up to max_total_nodes for zero-reallocation speed
        self.rates = np.zeros(self.max_total_nodes, dtype=np.float32)
        self.thresholds = np.zeros(self.max_total_nodes, dtype=np.float32)
        self.neuromod = np.zeros(4, dtype=np.float32)  # DA, 5HT, OA, ACh
        self.astrocyte_ca: float = 0.0
        self.last_pred = np.zeros(9, dtype=np.float32)

        # Input, readout, and prediction heads (rows/cols up to max_total_nodes)
        scale_in = 1.0 / np.sqrt(OBS_DIM)
        self.w_in = (self.rng.standard_normal((self.max_total_nodes, OBS_DIM)) * scale_in).astype(np.float32)
        # Mask sensory/visual neurons to receive stronger direct input
        sens_mask = np.isin(ref.roles, ["sensory", "visual"])
        self.w_in[:n][~sens_mask] *= 0.25

        scale_out = 1.0 / np.sqrt(max(1, n))
        self.w_out = (self.rng.standard_normal((ACTION_DIM, self.max_total_nodes)) * scale_out).astype(np.float32)
        motor_mask = np.isin(ref.roles, ["motor", "descending"])
        self.w_out[:, :n][:, ~motor_mask] *= 0.35

        # Predicts compact 8-dim outcome/state summary + expected reward (1-dim)
        self.w_pred = (self.rng.standard_normal((9, self.max_total_nodes)) * scale_out).astype(np.float32)

        # Secondary RNG for optional dendritic/delay extensions so default RNG sequence stays bit-identical
        aux_rng = np.random.default_rng(seed + 7919)
        self.max_delay = 4
        self.delay_ring = np.zeros((self.max_delay, self.max_total_nodes), dtype=np.float32)
        self._ring_ptr: int = 0
        self.w_apical = (aux_rng.standard_normal((self.max_total_nodes, 9)) * 0.08).astype(np.float32)

        # Optional multi-branch specialist readouts + context router (used by gated_specialist_routing fusion)
        self.specialist_w_outs: List[np.ndarray] = []
        self.router_w: Optional[np.ndarray] = None  # shape (K, 10): 6 task-onehot + 4 neuromod
        self.last_routing_weights: np.ndarray = np.ones(1, dtype=np.float32)

        # Eligibility traces for online plasticity
        self.elig_out = np.zeros_like(self.w_out)
        self.elig_bio = np.zeros_like(self._bio_weights)
        self.energy_used: float = 0.0
        self._innovation_counter: int = 0
        self._cached_csr: Optional[sparse.csr_matrix] = None

    @property
    def n_active(self) -> int:
        return self.ref.n_nodes + self.overlay.n_novel_neurons

    @property
    def bio_weights(self) -> np.ndarray:
        return self._bio_weights

    def _ensure_cow_weights(self) -> np.ndarray:
        if not self._bio_weights_owned:
            self._bio_weights = self._bio_weights.copy()
            self._bio_weights_owned = True
        return self._bio_weights

    def _invalidate_csr(self) -> None:
        self._cached_csr = None

    def _build_csr(self) -> sparse.csr_matrix:
        if self._cached_csr is not None:
            return self._cached_csr
        n_act = self.n_active
        # Biological edges minus pruned
        bio_w = self._bio_weights
        if self.overlay.pruned_bio_edges:
            bio_w = bio_w.copy()
            for idx in self.overlay.pruned_bio_edges:
                bio_w[idx] = 0.0
        if n_act > self.ref.n_nodes:
            pad = np.full(n_act - self.ref.n_nodes, self.ref.indptr[-1], dtype=np.int64)
            indptr = np.concatenate([self.ref.indptr, pad])
        else:
            indptr = self.ref.indptr
        bio_mat = sparse.csr_matrix(
            (bio_w, self.ref.indices, indptr),
            shape=(n_act, n_act),
            dtype=np.float32,
        )
        if self.overlay.novel_edges:
            rows = [k[0] for k in self.overlay.novel_edges.keys()]
            cols = [k[1] for k in self.overlay.novel_edges.keys()]
            vals = [v[0] for v in self.overlay.novel_edges.values()]
            nov_mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n_act, n_act), dtype=np.float32).tocsr()
            mat = (bio_mat + nov_mat).tocsr()
        else:
            mat = bio_mat
        self._cached_csr = mat
        return mat

    def node_sign(self, idx: int) -> int:
        if idx < self.ref.n_nodes:
            return int(self.ref.signs[idx])
        nov_idx = idx - self.ref.n_nodes
        return int(self.overlay.novel_signs[nov_idx])

    def node_role(self, idx: int) -> str:
        if idx < self.ref.n_nodes:
            return str(self.ref.roles[idx])
        nov_idx = idx - self.ref.n_nodes
        return str(self.overlay.novel_roles[nov_idx])

    def node_identity(self, idx: int) -> str:
        if idx < self.ref.n_nodes:
            return f"bio:{int(self.ref.root_ids[idx])}"
        nov_idx = idx - self.ref.n_nodes
        return f"nov:{self.overlay.novel_innovation_ids[nov_idx]}"

    def reset_state(self) -> None:
        self.rates.fill(0.0)
        self.elig_out.fill(0.0)
        self.elig_bio.fill(0.0)
        self.neuromod.fill(0.0)
        self.astrocyte_ca = 0.0
        self.last_pred.fill(0.0)
        self.delay_ring.fill(0.0)
        self._ring_ptr = 0

    def forward(
        self,
        obs: np.ndarray,
        external_guidance: Optional[np.ndarray] = None,
        explore_noise: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation tick; returns (action shape (5,), prediction shape (9,))."""
        n_act = self.n_active
        mat = self._build_csr()
        r_act = self.rates[:n_act]

        if self.enable_delays:
            # Multi-tap axonal delay line blending across 4 ticks (weights 0.45, 0.25, 0.20, 0.10)
            tap_weights = (0.45, 0.25, 0.20, 0.10)
            blended_r = np.zeros(n_act, dtype=np.float32)
            for lag, tw in enumerate(tap_weights):
                idx = (self._ring_ptr - lag) % self.max_delay
                blended_r += tw * self.delay_ring[idx, :n_act]
            rec_input = mat @ (0.5 * r_act + 0.5 * blended_r)
        else:
            rec_input = mat @ r_act

        ext_input = self.w_in[:n_act] @ obs

        # Two-compartment apical-basal dendritic gating (predictive top-down amplification)
        if self.enable_dendritic_gating:
            apical_gate = 0.7 + 0.6 / (1.0 + np.exp(-np.clip(self.w_apical[:n_act] @ self.last_pred, -6.0, 6.0)))
            ext_input = ext_input * apical_gate.astype(np.float32)

        # Compartment gains
        if self.overlay.compartment_gains:
            gains = np.ones(n_act, dtype=np.float32)
            for idx, g in self.overlay.compartment_gains.items():
                if idx < n_act:
                    gains[idx] = g
            rec_input = rec_input * gains

        # Neuromodulatory gain (bounded, stabilized by astrocytic calcium reservoir when enabled)
        astro_damp = (1.0 - 0.12 * self.astrocyte_ca) if (self.enable_delays or self.enable_dendritic_gating) else 1.0
        mod_gain = (1.0 + 0.15 * float(self.neuromod[0] + self.neuromod[3])) * astro_damp
        total = mod_gain * (rec_input + ext_input) - self.thresholds[:n_act]
        if external_guidance is not None:
            g_len = min(n_act, len(external_guidance))
            total[:g_len] += np.clip(external_guidance[:g_len], -0.5, 0.5)

        # Leaky rate dynamics with bounded firing rate
        new_r = np.tanh(0.6 * r_act + 0.4 * total)
        new_r = np.clip(new_r, -self.budgets.firing_rate_cap, self.budgets.firing_rate_cap)
        self.rates[:n_act] = new_r.astype(np.float32)

        # Update axonal delay ring & astrocytic calcium trace
        self._ring_ptr = (self._ring_ptr + 1) % self.max_delay
        self.delay_ring[self._ring_ptr, :n_act] = new_r
        mean_abs_r = float(np.mean(np.abs(new_r))) if n_act > 0 else 0.0
        self.astrocyte_ca = float(np.clip(0.92 * self.astrocyte_ca + 0.08 * max(0.0, mean_abs_r - 0.30) * 2.5, 0.0, 1.0))

        # Adaptive threshold homeostasis
        self.thresholds[:n_act] = np.clip(
            self.thresholds[:n_act] + 0.005 * (np.abs(new_r) - 0.25),
            -0.5,
            0.5,
        )

        # Energy proxy
        self.energy_used += float(np.sum(np.abs(new_r))) * 1e-4 + float(mat.nnz) * 1e-6

        if self.specialist_w_outs and self.router_w is not None:
            # Context-gated routing over preserved specialist parent readout branches
            ctx = np.concatenate([obs[143:149], self.neuromod]).astype(np.float32)  # 6 task-onehot + 4 neuromod = 10
            logits = self.router_w @ ctx
            logits = logits - np.max(logits)
            exp_l = np.exp(np.clip(logits, -12.0, 12.0))
            alphas = exp_l / np.maximum(1e-8, np.sum(exp_l))
            self.last_routing_weights = alphas.astype(np.float32)
            blended_act = np.zeros(ACTION_DIM, dtype=np.float32)
            for k_idx, w_sp in enumerate(self.specialist_w_outs):
                cols = min(n_act, w_sp.shape[1])
                blended_act += float(alphas[k_idx]) * (w_sp[:, :cols] @ new_r[:cols])
            action_raw = np.tanh(0.5 * (self.w_out[:, :n_act] @ new_r) + 0.5 * blended_act)
        else:
            action_raw = np.tanh(self.w_out[:, :n_act] @ new_r)

        if explore_noise > 0.0:
            action_raw = np.clip(action_raw + self.rng.normal(0.0, explore_noise, size=ACTION_DIM), -1.0, 1.0)
        pred = self.w_pred[:, :n_act] @ new_r
        self.last_pred = pred.astype(np.float32)
        return action_raw.astype(np.float32), pred.astype(np.float32)

    def update_plasticity(
        self,
        obs: np.ndarray,
        action_taken: np.ndarray,
        pred: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        lr: float = 0.02,
        update_recurrent: bool = True,
    ) -> float:
        """Reward-modulated eligibility trace plasticity + prediction error update."""
        n_act = self.n_active
        r_act = self.rates[:n_act]

        # Target summary: 8 proprioceptive dims (indices 135:143) + reward
        target_summary = np.concatenate([next_obs[135:143], [reward]]).astype(np.float32)
        pred_err_vec = target_summary - pred
        pred_err = float(np.mean(pred_err_vec ** 2))

        # Train prediction head
        self.w_pred[:, :n_act] += (0.5 * lr) * np.outer(pred_err_vec, r_act)
        np.clip(self.w_pred[:, :n_act], -2.0, 2.0, out=self.w_pred[:, :n_act])

        # Neuromodulator update: DA ~ reward - predicted_reward
        rpe = float(reward - pred[8])
        self.neuromod[0] = float(np.clip(0.8 * self.neuromod[0] + 0.2 * rpe, self.budgets.neuromod_min, self.budgets.neuromod_max))
        self.neuromod[3] = float(np.clip(0.8 * self.neuromod[3] + 0.2 * min(1.0, pred_err), self.budgets.neuromod_min, self.budgets.neuromod_max))

        # Eligibility trace for motor readout
        self.elig_out[:, :n_act] = 0.85 * self.elig_out[:, :n_act] + 0.15 * np.outer(action_taken, r_act)
        self.w_out[:, :n_act] += lr * np.clip(rpe, -1.0, 1.0) * self.elig_out[:, :n_act]
        np.clip(self.w_out[:, :n_act], -2.0, 2.0, out=self.w_out[:, :n_act])

        # Recurrent biological weight update (CoW + Dale's law sign preservation)
        if update_recurrent and abs(rpe) > 1e-4:
            w = self._ensure_cow_weights()
            pre_r = r_act[self.ref.indices]
            post_idx = np.repeat(np.arange(self.ref.n_nodes, dtype=np.int32), np.diff(self.ref.indptr))
            post_r = r_act[post_idx]
            self.elig_bio = 0.85 * self.elig_bio + 0.15 * (post_r * pre_r)
            delta = (0.25 * lr * np.clip(rpe, -1.0, 1.0)) * self.elig_bio
            w += delta
            # Enforce Dale's law: preserve sign of presynaptic neuron
            pre_signs = self.ref.signs[self.ref.indices].astype(np.float32)
            nonzero = pre_signs != 0
            w[nonzero] = pre_signs[nonzero] * np.clip(np.abs(w[nonzero]), 0.0, 1.5)
            w[~nonzero] = 0.0
            self._invalidate_csr()

        return pred_err

    # --- Structural developmental operations ---
    def _next_innovation_id(self, prefix: str) -> str:
        self._innovation_counter += 1
        return f"{self.lineage_id}_g{self.generation}_{prefix}_{self._innovation_counter}"

    def add_neuron(self, role: str = "central", sign: int = 1, birth_event: str = "neurogenesis", hypothesis: str = "expand capacity") -> Optional[int]:
        if self.overlay.n_novel_neurons >= self.budgets.max_new_neurons:
            return None
        new_idx = self.n_active
        inn_id = self._next_innovation_id("N")
        self.overlay.n_novel_neurons += 1
        self.overlay.novel_signs.append(int(sign))
        self.overlay.novel_roles.append(role)
        self.overlay.novel_innovation_ids.append(inn_id)
        self.overlay.innovations.append(
            InnovationRecord(
                innovation_id=inn_id,
                kind="neuron",
                parent_lineage=self.lineage_id,
                generation=self.generation,
                birth_event=birth_event,
                function_hypothesis=hypothesis,
                mutation_history=[f"born@{self.lineage_id}:g{self.generation}"],
            )
        )
        # Connect new neuron to top active sensory/central/motor nodes
        self.w_in[new_idx] = (self.rng.standard_normal(OBS_DIM) * 0.05).astype(np.float32)
        self.w_out[:, new_idx] = (self.rng.standard_normal(ACTION_DIM) * 0.05).astype(np.float32)
        self.w_pred[:, new_idx] = (self.rng.standard_normal(9) * 0.05).astype(np.float32)
        self._invalidate_csr()
        return new_idx

    def add_edge(
        self,
        post: int,
        pre: int,
        weight: Optional[float] = None,
        compartment: int = 0,
        birth_event: str = "synaptogenesis",
        hypothesis: str = "bridge active motifs",
    ) -> Optional[str]:
        if len(self.overlay.novel_edges) >= self.budgets.max_new_edges:
            return None
        if post >= self.n_active or pre >= self.n_active or post == pre:
            return None
        sign = self.node_sign(pre)
        if sign == 0:
            sign = 1
        w_val = float(sign * abs(weight)) if weight is not None else float(sign * self.rng.uniform(0.05, 0.25))
        inn_id = self._next_innovation_id("E")
        self.overlay.novel_edges[(post, pre)] = (w_val, inn_id, compartment)
        self.overlay.innovations.append(
            InnovationRecord(
                innovation_id=inn_id,
                kind="edge",
                parent_lineage=self.lineage_id,
                generation=self.generation,
                birth_event=birth_event,
                function_hypothesis=hypothesis,
                mutation_history=[f"syn@{post}<-{pre}"],
            )
        )
        self._invalidate_csr()
        return inn_id

    def prune_weak_edges(self, threshold: float = 0.01, max_prune: int = 16) -> int:
        pruned = 0
        w = self._bio_weights
        weak = np.where(np.abs(w) < threshold)[0]
        for idx in weak[:max_prune]:
            idx_i = int(idx)
            if idx_i not in self.overlay.pruned_bio_edges:
                self.overlay.pruned_bio_edges.add(idx_i)
                pruned += 1
        if pruned > 0:
            self._invalidate_csr()
        return pruned

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self._bio_weights.tobytes())
        h.update(self.w_in[:, :].tobytes())
        h.update(self.w_out[:, :].tobytes())
        return h.hexdigest()[:16]

    def snapshot(self) -> Dict[str, Any]:
        """Immutable deep snapshot for checkpointing & rollback."""
        return {
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "seed": self.seed,
            "enable_delays": self.enable_delays,
            "enable_dendritic_gating": self.enable_dendritic_gating,
            "bio_weights": self._bio_weights.copy(),
            "overlay": self.overlay.copy(),
            "w_in": self.w_in.copy(),
            "w_out": self.w_out.copy(),
            "w_pred": self.w_pred.copy(),
            "w_apical": self.w_apical.copy(),
            "specialist_w_outs": [w.copy() for w in self.specialist_w_outs],
            "router_w": self.router_w.copy() if self.router_w is not None else None,
            "thresholds": self.thresholds.copy(),
            "neuromod": self.neuromod.copy(),
            "astrocyte_ca": self.astrocyte_ca,
            "elig_out": self.elig_out.copy(),
            "elig_bio": self.elig_bio.copy(),
            "energy_used": self.energy_used,
            "innovation_counter": self._innovation_counter,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self.lineage_id = snap["lineage_id"]
        self.generation = snap["generation"]
        self.seed = snap["seed"]
        self.enable_delays = bool(snap.get("enable_delays", False))
        self.enable_dendritic_gating = bool(snap.get("enable_dendritic_gating", False))
        self._bio_weights = snap["bio_weights"].copy()
        self._bio_weights_owned = True
        self.overlay = snap["overlay"].copy()
        self.w_in = snap["w_in"].copy()
        self.w_out = snap["w_out"].copy()
        self.w_pred = snap["w_pred"].copy()
        if "w_apical" in snap:
            self.w_apical = snap["w_apical"].copy()
        self.specialist_w_outs = [w.copy() for w in snap.get("specialist_w_outs", [])]
        self.router_w = snap["router_w"].copy() if snap.get("router_w") is not None else None
        self.thresholds = snap["thresholds"].copy()
        self.neuromod = snap["neuromod"].copy()
        self.astrocyte_ca = float(snap.get("astrocyte_ca", 0.0))
        self.elig_out = snap["elig_out"].copy()
        self.elig_bio = snap["elig_bio"].copy()
        self.energy_used = float(snap["energy_used"])
        self._innovation_counter = int(snap["innovation_counter"])
        self.rng.bit_generator.state = copy.deepcopy(snap["rng_state"])
        self._invalidate_csr()

    def clone_child(self, child_lineage: str, child_gen: int, seed: int) -> "ConnectomeBrain":
        """Create a child brain sharing CoW weights initially."""
        child = ConnectomeBrain(
            ref=self.ref,
            lineage_id=child_lineage,
            generation=child_gen,
            seed=seed,
            budgets=self.budgets,
            shared_base_weights=self._bio_weights,
            enable_delays=self.enable_delays,
            enable_dendritic_gating=self.enable_dendritic_gating,
        )
        child.overlay = self.overlay.copy()
        child.w_in = self.w_in.copy()
        child.w_out = self.w_out.copy()
        child.w_pred = self.w_pred.copy()
        child.w_apical = self.w_apical.copy()
        child.specialist_w_outs = [w.copy() for w in self.specialist_w_outs]
        child.router_w = self.router_w.copy() if self.router_w is not None else None
        child.thresholds = self.thresholds.copy()
        child.neuromod = self.neuromod.copy()
        child.astrocyte_ca = self.astrocyte_ca
        child._innovation_counter = self._innovation_counter
        return child

    def export_graph_telemetry(self, max_bio_edges: int = 320) -> Dict[str, Any]:
        """Export compact JSON-serializable connectome topology, firing rates, and provenance for the GUI."""
        n_act = self.n_active
        n_bio = self.ref.n_nodes
        nodes = []
        for idx in range(n_act):
            is_nov = idx >= n_bio
            nodes.append({
                "idx": idx,
                "identity": self.node_identity(idx),
                "role": self.node_role(idx),
                "sign": self.node_sign(idx),
                "cell_type": "NOVEL_INTERNEURON" if is_nov else str(self.ref.cell_types[idx]),
                "is_novel": is_nov,
                "rate": round(float(self.rates[idx]), 4),
                "threshold": round(float(self.thresholds[idx]), 4),
                "compartment_gain": round(float(self.overlay.compartment_gains.get(idx, 1.0)), 3),
            })

        # Select top biological edges by |weight| + all novel/transplanted edges
        edges = []
        bio_w = self._bio_weights
        post_arr = np.repeat(np.arange(n_bio, dtype=np.int32), np.diff(self.ref.indptr))
        active_mask = np.ones(len(bio_w), dtype=bool)
        if self.overlay.pruned_bio_edges:
            for p_idx in self.overlay.pruned_bio_edges:
                if p_idx < len(active_mask):
                    active_mask[p_idx] = False
        cand_idx = np.where(active_mask & (np.abs(bio_w) > 1e-4))[0]
        if len(cand_idx) > max_bio_edges:
            top_order = np.argsort(np.abs(bio_w[cand_idx]))[::-1][:max_bio_edges]
            cand_idx = cand_idx[top_order]

        for e_i in cand_idx:
            edges.append({
                "pre": int(self.ref.indices[e_i]),
                "post": int(post_arr[e_i]),
                "weight": round(float(bio_w[e_i]), 4),
                "syn_count": int(self.ref.synapse_counts[e_i]),
                "is_novel": False,
            })

        for (post, pre), (w_val, inn_id, comp) in self.overlay.novel_edges.items():
            edges.append({
                "pre": int(pre),
                "post": int(post),
                "weight": round(float(w_val), 4),
                "syn_count": 1,
                "is_novel": True,
                "innovation_id": inn_id,
                "compartment": int(comp),
            })

        return {
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "n_bio_nodes": n_bio,
            "n_active_nodes": n_act,
            "n_novel_neurons": self.overlay.n_novel_neurons,
            "n_novel_edges": len(self.overlay.novel_edges),
            "n_pruned_edges": len(self.overlay.pruned_bio_edges),
            "weight_hash": self.weight_hash(),
            "neuromod": [round(float(x), 4) for x in self.neuromod],
            "astrocyte_ca": round(float(self.astrocyte_ca), 4),
            "routing_weights": [round(float(x), 4) for x in self.last_routing_weights],
            "energy_used": round(float(self.energy_used), 4),
            "nodes": nodes,
            "edges": edges,
            "innovations": [asdict(inn) for inn in self.overlay.innovations[-40:]],
        }


class RecurrentANNAnalogue:
    """Conventional dense/sparse recurrent ANN analogue with matched node count for comparison."""

    def __init__(self, hidden_dim: int = 256, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)
        self.h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_in = (self.rng.standard_normal((hidden_dim, OBS_DIM)) / np.sqrt(OBS_DIM)).astype(np.float32)
        self.w_rec = (self.rng.standard_normal((hidden_dim, hidden_dim)) * (0.8 / np.sqrt(hidden_dim))).astype(np.float32)
        self.w_out = (self.rng.standard_normal((ACTION_DIM, hidden_dim)) / np.sqrt(hidden_dim)).astype(np.float32)
        self.w_pred = (self.rng.standard_normal((9, hidden_dim)) / np.sqrt(hidden_dim)).astype(np.float32)
        self.elig_out = np.zeros_like(self.w_out)
        self.energy_used: float = 0.0

    def reset_state(self) -> None:
        self.h.fill(0.0)
        self.elig_out.fill(0.0)

    def forward(self, obs: np.ndarray, explore_noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        self.h = np.tanh(0.6 * self.h + 0.4 * (self.w_rec @ self.h + self.w_in @ obs)).astype(np.float32)
        self.energy_used += float(np.sum(np.abs(self.h))) * 1e-4 + float(self.hidden_dim ** 2) * 1e-6
        act = np.tanh(self.w_out @ self.h)
        if explore_noise > 0.0:
            act = np.clip(act + self.rng.normal(0.0, explore_noise, size=ACTION_DIM), -1.0, 1.0)
        pred = self.w_pred @ self.h
        return act.astype(np.float32), pred.astype(np.float32)

    def update_plasticity(self, obs: np.ndarray, action_taken: np.ndarray, pred: np.ndarray, next_obs: np.ndarray, reward: float, lr: float = 0.02) -> float:
        target = np.concatenate([next_obs[135:143], [reward]]).astype(np.float32)
        err_vec = target - pred
        pred_err = float(np.mean(err_vec ** 2))
        self.w_pred += (0.5 * lr) * np.outer(err_vec, self.h)
        rpe = float(np.clip(reward - pred[8], -1.0, 1.0))
        self.elig_out = 0.85 * self.elig_out + 0.15 * np.outer(action_taken, self.h)
        self.w_out += lr * rpe * self.elig_out
        np.clip(self.w_out, -2.0, 2.0, out=self.w_out)
        return pred_err

    def snapshot(self) -> Dict[str, Any]:
        return {
            "w_in": self.w_in.copy(),
            "w_rec": self.w_rec.copy(),
            "w_out": self.w_out.copy(),
            "w_pred": self.w_pred.copy(),
            "elig_out": self.elig_out.copy(),
            "energy_used": self.energy_used,
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self.w_in = snap["w_in"].copy()
        self.w_rec = snap["w_rec"].copy()
        self.w_out = snap["w_out"].copy()
        self.w_pred = snap["w_pred"].copy()
        self.elig_out = snap["elig_out"].copy()
        self.energy_used = float(snap["energy_used"])
