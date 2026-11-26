"""Ancestry-aware graph matching and exact-inheritance fusion engine."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evolution import ParentDiagnostics
from .substrate import ConnectomeBrain


@dataclass
class ProvenanceReport:
    strategy: str
    parent_lineages: List[str]
    homologous_matches: int
    novel_matches: int
    unmatched_novel_kept: int
    conflicts_recorded: int
    discarded_components: int
    exact_inheritance_verified: bool
    inherited_weight_hash: str
    node_provenance: Dict[int, str] = field(default_factory=dict)
    conflict_log: List[str] = field(default_factory=list)


def _novel_motif_signature(brain: ConnectomeBrain, node_idx: int) -> np.ndarray:
    """Compute 4-dim local structural motif vector [in_deg, out_deg, exc_ratio, reciprocity]."""
    in_edges = [w for (post, pre), (w, _, _) in brain.overlay.novel_edges.items() if post == node_idx]
    out_edges = [w for (post, pre), (w, _, _) in brain.overlay.novel_edges.items() if pre == node_idx]
    in_d = len(in_edges)
    out_d = len(out_edges)
    exc_in = sum(1 for w in in_edges if w > 0) / max(1, in_d)
    in_neighbors = {pre for (post, pre) in brain.overlay.novel_edges.keys() if post == node_idx}
    out_neighbors = {post for (post, pre) in brain.overlay.novel_edges.keys() if pre == node_idx}
    recip = len(in_neighbors & out_neighbors) / max(1, len(in_neighbors | out_neighbors))
    return np.array([min(1.0, in_d / 8.0), min(1.0, out_d / 8.0), exc_in, recip], dtype=np.float64)


def _match_novel_neurons(
    parent_a: ConnectomeBrain,
    diag_a: ParentDiagnostics,
    parent_b: ConnectomeBrain,
    diag_b: ParentDiagnostics,
    min_confidence: float = 0.55,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Confidence-weighted sparse assignment for novel neurons using role, sign, motif, and activation similarity."""
    n_bio = parent_a.ref.n_nodes
    nov_a = list(range(n_bio, parent_a.n_active))
    nov_b = list(range(n_bio, parent_b.n_active))
    if not nov_a or not nov_b:
        return [], nov_a, nov_b

    motifs_a = [_novel_motif_signature(parent_a, ia) for ia in nov_a]
    motifs_b = [_novel_motif_signature(parent_b, ib) for ib in nov_b]

    cost = np.ones((len(nov_a), len(nov_b)), dtype=np.float64)
    for i, ia in enumerate(nov_a):
        for j, ib in enumerate(nov_b):
            role_sim = 1.0 if parent_a.node_role(ia) == parent_b.node_role(ib) else 0.0
            sign_sim = 1.0 if parent_a.node_sign(ia) == parent_b.node_sign(ib) else 0.0
            act_a = float(diag_a.mean_activation_profile[ia]) if ia < len(diag_a.mean_activation_profile) else 0.0
            act_b = float(diag_b.mean_activation_profile[ib]) if ib < len(diag_b.mean_activation_profile) else 0.0
            act_sim = 1.0 - min(1.0, abs(act_a - act_b))
            motif_sim = 1.0 - min(1.0, float(np.mean(np.abs(motifs_a[i] - motifs_b[j]))))
            conf = 0.35 * role_sim + 0.25 * sign_sim + 0.25 * act_sim + 0.15 * motif_sim
            cost[i, j] = 1.0 - conf

    row_ind, col_ind = linear_sum_assignment(cost)
    matched: List[Tuple[int, int, float]] = []
    used_a, used_b = set(), set()
    for r, c in zip(row_ind, col_ind):
        conf = 1.0 - cost[r, c]
        if conf >= min_confidence:
            matched.append((nov_a[r], nov_b[c], float(conf)))
            used_a.add(nov_a[r])
            used_b.add(nov_b[c])

    unmatched_a = [ia for ia in nov_a if ia not in used_a]
    unmatched_b = [ib for ib in nov_b if ib not in used_b]
    return matched, unmatched_a, unmatched_b


def fuse_parents(
    parents: List[ConnectomeBrain],
    diagnostics: List[ParentDiagnostics],
    strategy: str = "hybrid_exact",
    seed: int = 42,
) -> Tuple[ConnectomeBrain, ProvenanceReport, Dict[int, Dict[int, float]]]:
    """Fuse multiple evolved parent brains into a candidate child architecture.

    Supported strategies:
    - best_parent
    - consensus_median
    - weighted_union
    - aligned_averaging
    - circuit_transplantation
    - hybrid_exact (bit-for-bit exact weight inheritance for selected parent edges + specialist branches)

    Returns (fused_brain, provenance_report, parent_to_fused_node_map).
    """
    assert len(parents) == len(diagnostics) and len(parents) >= 1
    order = np.argsort([d.mean_reward for d in diagnostics])[::-1]
    ranked_parents = [parents[i] for i in order]
    ranked_diags = [diagnostics[i] for i in order]
    best_p, best_d = ranked_parents[0], ranked_diags[0]

    fused = best_p.clone_child(child_lineage="FUSED", child_gen=best_p.generation + 1, seed=seed)
    fused._ensure_cow_weights()
    n_bio = best_p.ref.n_nodes
    parent_maps: Dict[int, Dict[int, float]] = {i: {n: float(1.0) for n in range(n_bio)} for i in range(len(parents))}
    node_prov: Dict[int, str] = {n: f"bio_homolog:{best_p.lineage_id}" for n in range(n_bio)}
    conflicts: List[str] = []
    novel_matches_total = 0
    unmatched_kept = 0
    discarded = 0

    if strategy == "best_parent":
        h = fused.weight_hash()
        return fused, ProvenanceReport(
            strategy=strategy,
            parent_lineages=[best_p.lineage_id],
            homologous_matches=n_bio,
            novel_matches=0,
            unmatched_novel_kept=best_p.overlay.n_novel_neurons,
            conflicts_recorded=0,
            discarded_components=0,
            exact_inheritance_verified=True,
            inherited_weight_hash=h,
            node_provenance=node_prov,
        ), parent_maps

    # Match and align secondary parents to best_p
    w_stack = [p.bio_weights for p in ranked_parents]
    rewards = np.array([max(0.01, d.mean_reward + 2.0) for d in ranked_diags], dtype=np.float64)
    weights_norm = rewards / np.sum(rewards)

    if strategy == "consensus_median":
        fused._bio_weights = np.median(np.stack(w_stack, axis=0), axis=0).astype(np.float32)
        fused.w_out[:, :n_bio] = np.median(np.stack([p.w_out[:, :n_bio] for p in ranked_parents], axis=0), axis=0).astype(np.float32)
    elif strategy == "aligned_averaging":
        avg_w = np.zeros_like(fused._bio_weights)
        avg_out = np.zeros_like(fused.w_out[:, :n_bio])
        for w_p, p, alpha in zip(w_stack, ranked_parents, weights_norm):
            avg_w += float(alpha) * w_p
            avg_out += float(alpha) * p.w_out[:, :n_bio]
        fused._bio_weights = avg_w.astype(np.float32)
        fused.w_out[:, :n_bio] = avg_out.astype(np.float32)
    elif strategy in ("weighted_union", "circuit_transplantation", "hybrid_exact", "gated_specialist_routing"):
        # For hybrid_exact & gated_specialist_routing: preserve best_p's biological weights bit-for-bit
        # on core edges; transplant high-sensitivity specialist edges/nodes from secondary parents!
        if strategy == "weighted_union":
            for i in range(len(fused._bio_weights)):
                vals = [w[i] for w in w_stack]
                fused._bio_weights[i] = np.float32(np.sum(np.array(vals) * weights_norm))
        else:
            for p_idx, (sec_p, sec_d) in enumerate(zip(ranked_parents[1:], ranked_diags[1:]), start=1):
                matched_nov, _, unm_sec = _match_novel_neurons(best_p, best_d, sec_p, sec_d)
                novel_matches_total += len(matched_nov)
                for ia, ib, conf in matched_nov:
                    parent_maps[int(order[p_idx])][ib] = float(ia)
                for ib in unm_sec[:4]:
                    if fused.n_active < fused.max_total_nodes:
                        new_idx = fused.add_neuron(
                            role=sec_p.node_role(ib),
                            sign=sec_p.node_sign(ib),
                            birth_event=f"transplant_from_{sec_p.lineage_id}",
                            hypothesis="preserve_specialist_circuit",
                        )
                        if new_idx is not None:
                            fused.w_in[new_idx] = sec_p.w_in[ib].copy()
                            fused.w_out[:, new_idx] = sec_p.w_out[:, ib].copy()
                            node_prov[new_idx] = f"transplant:{sec_p.lineage_id}:{sec_p.node_identity(ib)}"
                            parent_maps[int(order[p_idx])][ib] = float(new_idx)
                            unmatched_kept += 1
                    else:
                        discarded += 1

                for (post, pre), (w_val, inn_id, comp) in list(sec_p.overlay.novel_edges.items())[:8]:
                    if post < fused.n_active and pre < fused.n_active:
                        if (post, pre) in fused.overlay.novel_edges:
                            conflicts.append(f"edge({post},{pre})_kept_{best_p.lineage_id}_over_{sec_p.lineage_id}")
                        else:
                            fused.add_edge(post, pre, weight=w_val, compartment=comp, birth_event=f"transplant_{sec_p.lineage_id}")

            if strategy == "gated_specialist_routing":
                # Keep every parent's specialist readout matrix bit-for-bit alongside a context router
                from .env import TASKS
                k_parents = len(ranked_parents)
                fused.specialist_w_outs = [p.w_out.copy() for p in ranked_parents]
                router_w = np.zeros((k_parents, 10), dtype=np.float32)
                for k_i, d_k in enumerate(ranked_diags):
                    for t_i, t_name in enumerate(TASKS):
                        router_w[k_i, t_i] = float(d_k.per_task_reward.get(t_name, 0.0))
                fused.router_w = router_w
                fused.last_routing_weights = (np.ones(k_parents, dtype=np.float32) / k_parents)

        if strategy in ("hybrid_exact", "gated_specialist_routing"):
            # Verify bit-for-bit preservation of inherited core biological weights
            exact_ok = np.array_equal(fused._bio_weights, best_p.bio_weights)
        else:
            exact_ok = False
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy!r}")

    fused._invalidate_csr()
    return fused, ProvenanceReport(
        strategy=strategy,
        parent_lineages=[p.lineage_id for p in ranked_parents],
        homologous_matches=n_bio,
        novel_matches=novel_matches_total,
        unmatched_novel_kept=unmatched_kept,
        conflicts_recorded=len(conflicts),
        discarded_components=discarded,
        exact_inheritance_verified=exact_ok if strategy in ("hybrid_exact", "gated_specialist_routing", "best_parent") else False,
        inherited_weight_hash=fused.weight_hash(),
        node_provenance=node_prov,
        conflict_log=conflicts,
    ), parent_maps
