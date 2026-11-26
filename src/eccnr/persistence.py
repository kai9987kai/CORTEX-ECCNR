"""Immutable disk checkpoints (.npz + SHA-256 manifest) and append-only supervision/lineage journals."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .connectome import ConnectomeReference
from .substrate import ConnectomeBrain, InnovationRecord, StructuralOverlay, SubstrateBudgets


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_brain_checkpoint(
    brain: ConnectomeBrain,
    ckpt_dir: Path,
    tag: str = "checkpoint",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save an immutable compressed .npz + .json checkpoint with SHA-256 verification."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{brain.lineage_id}_g{brain.generation}_{tag}"
    npz_path = ckpt_dir / f"{stem}.npz"
    meta_path = ckpt_dir / f"{stem}.json"

    arrays: Dict[str, Any] = {
        "bio_weights": brain.bio_weights,
        "w_in": brain.w_in,
        "w_out": brain.w_out,
        "w_pred": brain.w_pred,
        "w_apical": brain.w_apical,
        "thresholds": brain.thresholds,
        "neuromod": brain.neuromod,
        "elig_out": brain.elig_out,
        "elig_bio": brain.elig_bio,
    }
    if brain.router_w is not None:
        arrays["router_w"] = brain.router_w
    for idx, sp_w in enumerate(brain.specialist_w_outs):
        arrays[f"spec_w_{idx}"] = sp_w

    np.savez_compressed(npz_path, **arrays)
    npz_sha = _sha256_file(npz_path)

    overlay_payload = {
        "n_novel_neurons": brain.overlay.n_novel_neurons,
        "novel_signs": brain.overlay.novel_signs,
        "novel_roles": brain.overlay.novel_roles,
        "novel_innovation_ids": brain.overlay.novel_innovation_ids,
        "novel_edges": [
            {"post": k[0], "pre": k[1], "weight": v[0], "innovation_id": v[1], "compartment": v[2]}
            for k, v in brain.overlay.novel_edges.items()
        ],
        "pruned_bio_edges": sorted(brain.overlay.pruned_bio_edges),
        "compartment_gains": {str(k): v for k, v in brain.overlay.compartment_gains.items()},
        "innovations": [asdict(inn) for inn in brain.overlay.innovations],
    }

    manifest = {
        "checkpoint_id": stem,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": brain.ref.dataset_id,
        "topology_hash": brain.ref.topology_hash(),
        "lineage_id": brain.lineage_id,
        "generation": brain.generation,
        "seed": brain.seed,
        "enable_delays": brain.enable_delays,
        "enable_dendritic_gating": brain.enable_dendritic_gating,
        "astrocyte_ca": brain.astrocyte_ca,
        "energy_used": brain.energy_used,
        "n_specialists": len(brain.specialist_w_outs),
        "weight_hash": brain.weight_hash(),
        "npz_file": npz_path.name,
        "npz_sha256": npz_sha,
        "overlay": overlay_payload,
        "extra_meta": extra_meta or {},
    }
    meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_brain_checkpoint(
    meta_path: Path,
    ref: ConnectomeReference,
    budgets: Optional[SubstrateBudgets] = None,
) -> ConnectomeBrain:
    """Load and cryptographically verify a saved ConnectomeBrain checkpoint."""
    meta_path = Path(meta_path)
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    npz_path = meta_path.parent / manifest["npz_file"]
    actual_sha = _sha256_file(npz_path)
    if actual_sha != manifest["npz_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {npz_path}: expected {manifest['npz_sha256']}, got {actual_sha}")

    brain = ConnectomeBrain(
        ref=ref,
        lineage_id=manifest["lineage_id"],
        generation=int(manifest["generation"]),
        seed=int(manifest["seed"]),
        budgets=budgets,
        enable_delays=bool(manifest.get("enable_delays", False)),
        enable_dendritic_gating=bool(manifest.get("enable_dendritic_gating", False)),
    )
    with np.load(npz_path) as data:
        brain._bio_weights = data["bio_weights"].copy()
        brain._bio_weights_owned = True
        brain.w_in[:] = data["w_in"]
        brain.w_out[:] = data["w_out"]
        brain.w_pred[:] = data["w_pred"]
        if "w_apical" in data:
            brain.w_apical[:] = data["w_apical"]
        brain.thresholds[:] = data["thresholds"]
        brain.neuromod[:] = data["neuromod"]
        brain.elig_out[:] = data["elig_out"]
        brain.elig_bio[:] = data["elig_bio"]
        if "router_w" in data:
            brain.router_w = data["router_w"].copy()
        n_spec = int(manifest.get("n_specialists", 0))
        brain.specialist_w_outs = [data[f"spec_w_{i}"].copy() for i in range(n_spec)]

    ov = manifest["overlay"]
    brain.overlay = StructuralOverlay(
        n_novel_neurons=int(ov["n_novel_neurons"]),
        novel_signs=[int(x) for x in ov["novel_signs"]],
        novel_roles=[str(x) for x in ov["novel_roles"]],
        novel_innovation_ids=[str(x) for x in ov["novel_innovation_ids"]],
        novel_edges={
            (int(e["post"]), int(e["pre"])): (float(e["weight"]), str(e["innovation_id"]), int(e["compartment"]))
            for e in ov["novel_edges"]
        },
        pruned_bio_edges={int(x) for x in ov["pruned_bio_edges"]},
        compartment_gains={int(k): float(v) for k, v in ov["compartment_gains"].items()},
        innovations=[InnovationRecord(**inn) for inn in ov["innovations"]],
    )
    brain.astrocyte_ca = float(manifest.get("astrocyte_ca", 0.0))
    brain.energy_used = float(manifest.get("energy_used", 0.0))
    brain._invalidate_csr()
    return brain


class AppendOnlyJournal:
    """Append-only JSONL ledger for lineage provenance and bounded human/AI interventions.

    Maintains strict separation between automatic ratings and genuine supervisor decisions.
    """

    def __init__(self, journal_path: Path):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "event_id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            **payload,
        }
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def log_intervention(
        self,
        checkpoint_id: str,
        observation_summary: Dict[str, float],
        prediction_summary: Dict[str, float],
        automatic_judgement: str,
        supervisor_source: str,  # "human_gui" | "antigravity_mcp" | "automatic_only"
        supervisor_decision: str,  # "accept" | "reject" | "relabel" | "rollback" | "bounded_edit"
        rationale: str,
        intervention_applied: str,
    ) -> Dict[str, Any]:
        if supervisor_source not in ("human_gui", "antigravity_mcp", "automatic_only"):
            raise ValueError(f"Invalid supervisor_source: {supervisor_source}")
        return self.append(
            "supervision_intervention",
            {
                "checkpoint_id": checkpoint_id,
                "observation_summary": observation_summary,
                "prediction_summary": prediction_summary,
                "automatic_judgement": automatic_judgement,
                "supervisor_source": supervisor_source,
                "genuinely_reviewed": supervisor_source in ("human_gui", "antigravity_mcp"),
                "supervisor_decision": supervisor_decision,
                "rationale": rationale,
                "intervention_applied": intervention_applied,
            },
        )

    def read_all(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]
