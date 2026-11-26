"""FlyWire FAFB v783 acquisition, verification, sparse representation, and subgraph extraction.

Distinguishes measured anatomical synapse counts from simulated trainable weights.
Preserves original biological root IDs and topology as an immutable reference.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import sparse

FLYWIRE_RELEASE = {
    "dataset_id": "flywire_fafb_v783",
    "name": "FlyWire FAFB v783 Adult Female Drosophila Whole-Brain Connectome",
    "version": "v783 (annotations v2.1.0)",
    "license": "CC-BY-NC-4.0 (FAFB v783); CC-BY-4.0 (flywire_annotations v2.1.0)",
    "citations": [
        "Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. Nature 634, 124-138. https://doi.org/10.1038/s41586-024-07558-y",
        "Schlegel et al. (2024). Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634, 139-152. https://doi.org/10.1038/s41586-024-07686-5",
    ],
    "zenodo_record": "https://zenodo.org/records/10676866",
    "files": {
        "weights": {
            "filename": "proofread_connections_783.feather",
            "url": "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
            "md5": "f48f972d262323a102aed49af1396b8a",
        },
        "neurons": {
            "filename": "proofread_root_ids_783.npy",
            "url": "https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1",
            "md5": "e0e6c19732fd8c7a4e39a2d170105421",
        },
        "annotations": {
            "filename": "annotations_v2.1.0.tsv",
            "url": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            "sha256": "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6",
        },
    },
}

# Explicit modelling assumption for fast synaptic signs (Dale's law)
NT_SIGNS: Dict[str, int] = {
    "ach": 1,
    "acetylcholine": 1,
    "gaba": -1,
    "glut": -1,
    "glutamate": -1,
    "hist": -1,
    "histamine": -1,
    "da": 0,
    "dopamine": 0,
    "ser": 0,
    "serotonin": 0,
    "oct": 0,
    "octopamine": 0,
}

ROLE_ORDER = ("sensory", "visual", "central", "descending", "ascending", "motor", "unknown")


def _file_hashes(path: Path) -> Dict[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _verify_file(path: Path, spec: Dict[str, Any]) -> Dict[str, str]:
    hashes = _file_hashes(path)
    if "md5" in spec and hashes["md5"] != spec["md5"]:
        raise ValueError(f"MD5 mismatch for {path.name}: expected {spec['md5']}, got {hashes['md5']}")
    if "sha256" in spec and hashes["sha256"] != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path.name}: expected {spec['sha256']}, got {hashes['sha256']}")
    return hashes


def download_flywire(data_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Download and verify official FlyWire FAFB v783 files into data_dir/raw."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(raw_dir).free
    if free_bytes < 2 * (1024**3):
        raise RuntimeError(f"Insufficient disk space ({free_bytes / 1e9:.2f} GB free, need >= 2.0 GB)")

    verified: Dict[str, Any] = {}
    for key, spec in FLYWIRE_RELEASE["files"].items():
        dest = raw_dir / spec["filename"]
        if dest.exists() and not force:
            try:
                hashes = _verify_file(dest, spec)
                verified[key] = {"path": str(dest), "bytes": dest.stat().st_size, **hashes}
                continue
            except ValueError:
                pass
        tmp = raw_dir / f".{spec['filename']}.part"
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "CORTEX-ECCNR/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        hashes = _verify_file(tmp, spec)
        tmp.replace(dest)
        verified[key] = {"path": str(dest), "bytes": dest.stat().st_size, **hashes}
    return verified


def _map_super_class_to_role(super_class: str) -> str:
    sc = str(super_class).strip().lower()
    if "sensory" in sc:
        return "sensory"
    if "optic" in sc or "visual" in sc:
        return "visual"
    if "descending" in sc:
        return "descending"
    if "ascending" in sc:
        return "ascending"
    if "motor" in sc:
        return "motor"
    if "central" in sc or "endocrine" in sc:
        return "central"
    return "unknown"


def prepare_flywire_cache(data_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Build memory-mappable sparse CSR cache + manifest from verified FlyWire raw files."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    cache_dir = data_dir / "cache"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    verified = download_flywire(data_dir, force=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    root_ids = np.load(raw_dir / FLYWIRE_RELEASE["files"]["neurons"]["filename"])
    root_ids = np.sort(np.unique(root_ids.astype(np.int64)))
    n_nodes = int(root_ids.shape[0])

    ann_df = pd.read_csv(
        raw_dir / FLYWIRE_RELEASE["files"]["annotations"]["filename"],
        sep="\t",
        low_memory=False,
        usecols=lambda c: c in {"root_id", "super_class", "cell_type", "top_nt", "flow", "side"},
    )
    ann_df = ann_df.dropna(subset=["root_id"]).drop_duplicates(subset=["root_id"])
    ann_df["root_id"] = ann_df["root_id"].astype(np.int64)
    ann_map = ann_df.set_index("root_id")

    signs = np.zeros(n_nodes, dtype=np.int8)
    roles = np.full(n_nodes, "unknown", dtype="U12")
    cell_types = np.full(n_nodes, "", dtype="U40")
    annotated_count = 0
    nt_annotated_count = 0

    for idx, rid in enumerate(root_ids):
        if rid in ann_map.index:
            annotated_count += 1
            row = ann_map.loc[rid]
            sc = row["super_class"] if "super_class" in row and pd.notna(row["super_class"]) else ""
            roles[idx] = _map_super_class_to_role(sc)
            ct = row["cell_type"] if "cell_type" in row and pd.notna(row["cell_type"]) else ""
            cell_types[idx] = str(ct)[:40]
            nt = str(row["top_nt"]).strip().lower() if "top_nt" in row and pd.notna(row["top_nt"]) else ""
            if nt in NT_SIGNS:
                nt_annotated_count += 1
                signs[idx] = NT_SIGNS[nt]

    # Read connections feather and aggregate by (pre_pt_root_id, post_pt_root_id) across neuropils
    conn_tbl = feather.read_table(
        raw_dir / FLYWIRE_RELEASE["files"]["weights"]["filename"],
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
    )
    pre_raw = conn_tbl["pre_pt_root_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    post_raw = conn_tbl["post_pt_root_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    syn_raw = conn_tbl["syn_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    raw_contact_rows = int(len(syn_raw))
    total_synapses = int(syn_raw.sum())

    pre_idx = np.searchsorted(root_ids, pre_raw)
    post_idx = np.searchsorted(root_ids, post_raw)
    valid = (
        (pre_idx < n_nodes)
        & (post_idx < n_nodes)
        & (root_ids[np.minimum(pre_idx, n_nodes - 1)] == pre_raw)
        & (root_ids[np.minimum(post_idx, n_nodes - 1)] == post_raw)
    )
    pre_idx = pre_idx[valid].astype(np.int32)
    post_idx = post_idx[valid].astype(np.int32)
    syn_valid = syn_raw[valid].astype(np.int64)

    # Aggregate neuropil-split contacts into unique directed edges (pre -> post)
    # Build sparse CSR with rows=post, cols=pre for fast W @ x rate dynamics
    mat = sparse.coo_matrix((syn_valid, (post_idx, pre_idx)), shape=(n_nodes, n_nodes), dtype=np.int64).tocsr()
    mat.sum_duplicates()

    np.save(cache_dir / "root_ids.npy", root_ids)
    np.save(cache_dir / "signs.npy", signs)
    np.save(cache_dir / "roles.npy", roles)
    np.save(cache_dir / "cell_types.npy", cell_types)
    np.save(cache_dir / "csr_indptr.npy", mat.indptr.astype(np.int64))
    np.save(cache_dir / "csr_indices.npy", mat.indices.astype(np.int32))
    np.save(cache_dir / "csr_synapses.npy", mat.data.astype(np.int32))

    role_counts = {r: int(np.sum(roles == r)) for r in ROLE_ORDER}
    manifest = {
        "dataset_id": FLYWIRE_RELEASE["dataset_id"],
        "name": FLYWIRE_RELEASE["name"],
        "version": FLYWIRE_RELEASE["version"],
        "license": FLYWIRE_RELEASE["license"],
        "citations": FLYWIRE_RELEASE["citations"],
        "provenance_tier": "authentic",
        "node_count": n_nodes,
        "raw_neuropil_edge_rows": raw_contact_rows,
        "directed_edge_count": int(mat.nnz),
        "total_synapses": total_synapses,
        "synapse_multiplicity_min": int(mat.data.min()) if mat.nnz > 0 else 0,
        "synapse_multiplicity_median": float(np.median(mat.data)) if mat.nnz > 0 else 0.0,
        "synapse_multiplicity_max": int(mat.data.max()) if mat.nnz > 0 else 0,
        "annotation_coverage": {
            "annotated_nodes": annotated_count,
            "annotation_fraction": round(annotated_count / max(1, n_nodes), 4),
            "nt_annotated_nodes": nt_annotated_count,
            "nt_fraction": round(nt_annotated_count / max(1, n_nodes), 4),
        },
        "role_counts": role_counts,
        "source_hashes": verified,
        "aggregation_note": "Individual neuropil-split contact rows aggregated into unique directed edges (pre->post) summing syn_count; full connectivity retained.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class ConnectomeReference:
    """Immutable biological reference graph (sparse CSR: rows=post, cols=pre)."""

    dataset_id: str
    provenance_tier: str
    root_ids: np.ndarray          # int64, shape (N,)
    signs: np.ndarray             # int8, shape (N,), in {-1, 0, +1}
    roles: np.ndarray             # str, shape (N,)
    cell_types: np.ndarray        # str, shape (N,)
    indptr: np.ndarray            # int64, shape (N+1,)
    indices: np.ndarray           # int32, shape (E,) (presynaptic node index)
    synapse_counts: np.ndarray    # int32, shape (E,) (measured anatomical counts)
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.root_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    def to_initial_weights(self, row_norm_cap: float = 0.90) -> np.ndarray:
        """Convert anatomical synapse counts into simulated initial weights via signed log1p + row cap."""
        pre_signs = self.signs[self.indices].astype(np.float32)
        # Neurons with unknown/modulatory sign (0) get weak excitatory baseline +0.5 if signed needed,
        # or 0.0 for strict fast-synapse convention; here we keep strict 0.0 for modulatory/unknown
        # unless fallback is requested, matching Sonapse SIGN_CONVENTION.
        raw_w = pre_signs * np.log1p(self.synapse_counts.astype(np.float32))
        w = raw_w.copy()
        for post in range(self.n_nodes):
            s, e = int(self.indptr[post]), int(self.indptr[post + 1])
            if e > s:
                row_abs = float(np.sum(np.abs(w[s:e])))
                if row_abs > row_norm_cap and row_abs > 1e-8:
                    w[s:e] *= row_norm_cap / row_abs
        return w.astype(np.float32)

    def topology_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.root_ids.tobytes())
        h.update(self.indptr.tobytes())
        h.update(self.indices.tobytes())
        h.update(self.synapse_counts.tobytes())
        return h.hexdigest()[:16]


def load_flywire_subgraph(
    data_dir: Path,
    n_nodes: int = 256,
    seed: int = 42,
    min_synapse_count: int = 1,
) -> ConnectomeReference:
    """Load a role-stratified induced subgraph from the verified FlyWire FAFB v783 cache."""
    cache_dir = Path(data_dir) / "cache"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"FlyWire cache not found at {cache_dir}. Run scripts/acquire_flywire.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    root_ids = np.load(cache_dir / "root_ids.npy", mmap_mode="r")
    signs = np.load(cache_dir / "signs.npy", mmap_mode="r")
    roles = np.load(cache_dir / "roles.npy", mmap_mode="r")
    cell_types = np.load(cache_dir / "cell_types.npy", mmap_mode="r")
    indptr = np.load(cache_dir / "csr_indptr.npy", mmap_mode="r")
    indices = np.load(cache_dir / "csr_indices.npy", mmap_mode="r")
    synapses = np.load(cache_dir / "csr_synapses.npy", mmap_mode="r")

    total_n = len(root_ids)
    in_deg = np.diff(indptr)
    # Role-stratified quotas proportional to brain roles, ensuring sensory/visual/motor/central representation
    rng = np.random.default_rng(seed)
    quotas = {
        "sensory": max(16, int(0.12 * n_nodes)),
        "visual": max(16, int(0.14 * n_nodes)),
        "central": max(64, int(0.50 * n_nodes)),
        "descending": max(8, int(0.10 * n_nodes)),
        "ascending": max(8, int(0.08 * n_nodes)),
        "motor": max(4, int(0.06 * n_nodes)),
    }
    chosen: List[int] = []
    chosen_set = set()
    for role, q in quotas.items():
        mask = np.where((roles == role) & (signs != 0) & (in_deg > 0))[0]
        if len(mask) == 0:
            mask = np.where(roles == role)[0]
        if len(mask) > 0:
            # Sample from top-degree 4*q pool for dense functional connectivity
            top_pool = mask[np.argsort(in_deg[mask])[::-1][: min(len(mask), 4 * q)]]
            pick = rng.choice(top_pool, size=min(q, len(top_pool)), replace=False)
            for idx in pick:
                if int(idx) not in chosen_set:
                    chosen.append(int(idx))
                    chosen_set.add(int(idx))

    # Fill remainder from top-degree signed central/known neurons
    if len(chosen) < n_nodes:
        rem_pool = np.where((signs != 0) & (in_deg > 0))[0]
        top_rem = rem_pool[np.argsort(in_deg[rem_pool])[::-1][: min(len(rem_pool), 4 * n_nodes)]]
        for idx in rng.permutation(top_rem):
            if int(idx) not in chosen_set:
                chosen.append(int(idx))
                chosen_set.add(int(idx))
                if len(chosen) >= n_nodes:
                    break

    selected = np.array(sorted(chosen[:n_nodes]), dtype=np.int64)
    # Induced subgraph extraction
    sub_indptr = np.zeros(len(selected) + 1, dtype=np.int64)
    sub_indices_list: List[np.ndarray] = []
    sub_syn_list: List[np.ndarray] = []

    for new_post, old_post in enumerate(selected):
        s, e = int(indptr[old_post]), int(indptr[old_post + 1])
        if e > s:
            cols = np.asarray(indices[s:e])
            syns = np.asarray(synapses[s:e])
            pos = np.searchsorted(selected, cols)
            in_bounds = pos < len(selected)
            matched = np.zeros(len(cols), dtype=bool)
            matched[in_bounds] = (selected[pos[in_bounds]] == cols[in_bounds]) & (syns[in_bounds] >= min_synapse_count)
            if np.any(matched):
                sub_indices_list.append(pos[matched].astype(np.int32))
                sub_syn_list.append(syns[matched].astype(np.int32))
                sub_indptr[new_post + 1] = sub_indptr[new_post] + int(np.sum(matched))
            else:
                sub_indptr[new_post + 1] = sub_indptr[new_post]
        else:
            sub_indptr[new_post + 1] = sub_indptr[new_post]

    sub_indices = np.concatenate(sub_indices_list) if sub_indices_list else np.zeros(0, dtype=np.int32)
    sub_syns = np.concatenate(sub_syn_list) if sub_syn_list else np.zeros(0, dtype=np.int32)

    sub_manifest = {
        **manifest,
        "subgraph_extraction": {
            "seed": seed,
            "requested_nodes": n_nodes,
            "actual_nodes": int(len(selected)),
            "induced_edges": int(len(sub_indices)),
            "total_induced_synapses": int(sub_syns.sum()),
            "min_synapse_count": min_synapse_count,
        },
    }
    return ConnectomeReference(
        dataset_id=f"flywire_fafb_v783_sub{len(selected)}",
        provenance_tier="authentic_subgraph",
        root_ids=np.array(root_ids[selected], dtype=np.int64),
        signs=np.array(signs[selected], dtype=np.int8),
        roles=np.array(roles[selected], dtype="U12"),
        cell_types=np.array(cell_types[selected], dtype="U40"),
        indptr=sub_indptr,
        indices=sub_indices,
        synapse_counts=sub_syns,
        manifest=sub_manifest,
    )


def build_synthetic_connectome(n_nodes: int = 64, seed: int = 42, density: float = 0.12) -> ConnectomeReference:
    """Offline synthetic connectome fixture for unit tests (never labeled authentic)."""
    rng = np.random.default_rng(seed)
    root_ids = np.arange(720575940000000000, 720575940000000000 + n_nodes, dtype=np.int64)
    signs = rng.choice([-1, 1], size=n_nodes, p=[0.35, 0.65]).astype(np.int8)
    role_choices = ["sensory", "visual", "central", "descending", "ascending", "motor"]
    role_probs = [0.15, 0.15, 0.45, 0.10, 0.08, 0.07]
    roles = rng.choice(role_choices, size=n_nodes, p=role_probs).astype("U12")
    cell_types = np.array([f"SYN_{r[:3].upper()}_{i}" for i, r in enumerate(roles)], dtype="U40")

    mat = sparse.random(n_nodes, n_nodes, density=density, random_state=seed, format="csr")
    mat.setdiag(0)
    mat.eliminate_zeros()
    syns = rng.integers(1, 25, size=mat.nnz, dtype=np.int32)

    return ConnectomeReference(
        dataset_id=f"synthetic_fixture_{n_nodes}",
        provenance_tier="synthetic_fixture",
        root_ids=root_ids,
        signs=signs,
        roles=roles,
        cell_types=cell_types,
        indptr=mat.indptr.astype(np.int64),
        indices=mat.indices.astype(np.int32),
        synapse_counts=syns,
        manifest={"provenance_tier": "synthetic_fixture", "seed": seed, "n_nodes": n_nodes, "n_edges": int(mat.nnz)},
    )


def rewire_connectome_degree_preserving(ref: ConnectomeReference, seed: int = 101, swaps_per_edge: int = 4) -> ConnectomeReference:
    """Create a degree-preserving rewired graph control (preserves in/out degree & synapse multiset)."""
    rng = np.random.default_rng(seed)
    n = ref.n_nodes
    # Extract edge list (post, pre, syn)
    posts = np.repeat(np.arange(n, dtype=np.int32), np.diff(ref.indptr))
    pres = ref.indices.copy()
    syns = ref.synapse_counts.copy()
    e = len(pres)
    if e < 4:
        return ref

    edge_set = set(zip(posts.tolist(), pres.tolist()))
    n_swaps = e * swaps_per_edge
    for _ in range(n_swaps):
        i, j = int(rng.integers(0, e)), int(rng.integers(0, e))
        if i == j:
            continue
        u1, v1 = int(posts[i]), int(pres[i])
        u2, v2 = int(posts[j]), int(pres[j])
        if u1 == v2 or u2 == v1 or u1 == u2 or v1 == v2:
            continue
        if (u1, v2) in edge_set or (u2, v1) in edge_set:
            continue
        edge_set.remove((u1, v1))
        edge_set.remove((u2, v2))
        edge_set.add((u1, v2))
        edge_set.add((u2, v1))
        pres[i], pres[j] = v2, v1

    mat = sparse.coo_matrix((syns, (posts, pres)), shape=(n, n), dtype=np.int32).tocsr()
    return ConnectomeReference(
        dataset_id=f"{ref.dataset_id}_rewired",
        provenance_tier="rewired_control",
        root_ids=ref.root_ids.copy(),
        signs=ref.signs.copy(),
        roles=ref.roles.copy(),
        cell_types=ref.cell_types.copy(),
        indptr=mat.indptr.astype(np.int64),
        indices=mat.indices.astype(np.int32),
        synapse_counts=mat.data.astype(np.int32),
        manifest={**ref.manifest, "rewired_control_seed": seed},
    )
