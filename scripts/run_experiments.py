"""CLI runner for smoke test, 4x3 pilot, primary matched-budget experiments, and report generation."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eccnr.connectome import build_synthetic_connectome, load_flywire_subgraph
from eccnr.experiments import run_single_condition, run_successive_action_experiment

PRIMARY_CONDITIONS = [
    "fixed_connectome",
    "single_plastic",
    "sequential_correction",
    "parallel_best",
    "fusion_only",
    "fusion_ordinary_replay",
    "fusion_activity_replay",
    "full_dtr_finetune",
    "ann_analogue",
    "rewired_control",
]


def _ci95(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))


def run_suite(mode: str, data_dir: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    if mode == "smoke":
        ref = build_synthetic_connectome(n_nodes=48, seed=42)
        conditions = ["fixed_connectome", "single_plastic", "fusion_only", "full_dtr_finetune", "ann_analogue"]
        replicates = [42]
        n_lin, n_gen, budget, max_ep = 2, 2, 800, 30
    elif mode == "pilot":
        ref = load_flywire_subgraph(data_dir=data_dir, n_nodes=256, seed=42)
        conditions = PRIMARY_CONDITIONS
        replicates = [101, 202]
        n_lin, n_gen, budget, max_ep = 4, 3, 3200, 60
    elif mode == "primary":
        ref = load_flywire_subgraph(data_dir=data_dir, n_nodes=256, seed=42)
        conditions = PRIMARY_CONDITIONS
        replicates = [101, 202, 303, 404, 505]
        n_lin, n_gen, budget, max_ep = 4, 3, 4800, 60
        trainer = "eligibility"
    elif mode == "bptt_primary":
        ref = load_flywire_subgraph(data_dir=data_dir, n_nodes=256, seed=42)
        conditions = PRIMARY_CONDITIONS
        replicates = [101, 202, 303]
        n_lin, n_gen, budget, max_ep = 4, 3, 24000, 60
        trainer = "bptt"
    elif mode == "advance_verify":
        ref = load_flywire_subgraph(data_dir=data_dir, n_nodes=256, seed=42)
        conditions = [
            "single_plastic",
            "full_dtr_finetune",
            "fusion_ordinary_replay",
            "ewc_dtr_finetune",
            "gated_fusion_ewc",
        ]
        replicates = [101, 202, 303]
        n_lin, n_gen, budget, max_ep = 4, 3, 12000, 60
        trainer = "bptt"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode in ("smoke", "pilot"):
        trainer = "eligibility"

    train_seeds_base = [10, 11, 12, 13]
    val_seeds_base = [20, 21, 22]
    sealed_seeds = [9001, 9002, 9003, 9004, 9005]  # Held-out sealed test seeds

    raw_rows: List[Dict[str, Any]] = []
    for cond in conditions:
        for rep_seed in replicates:
            res = run_single_condition(
                condition=cond,
                ref=ref,
                train_seeds=[s + rep_seed for s in train_seeds_base],
                val_seeds=[s + rep_seed for s in val_seeds_base],
                sealed_seeds=sealed_seeds,
                total_env_budget=budget,
                n_lineages=n_lin,
                n_generations=n_gen,
                max_episode_steps=max_ep,
                seed=rep_seed,
                trainer=trainer,
            )
            raw_rows.append(res)
            print(
                f"[{mode}] {cond:24s} seed={rep_seed} reward={res['sealed_mean_reward']:+.4f} "
                f"succ={res['sealed_success_rate']:.3f} steps={res['env_steps_used']} ({res['wall_seconds']}s)",
                flush=True,
            )

    succ_action = run_successive_action_experiment(ref, seeds=[11, 12, 9001, 9002, 9003], max_episode_steps=max_ep)

    df = pd.DataFrame(raw_rows)
    df.to_csv(out_dir / f"{mode}_raw_results.csv", index=False)

    summary_rows = []
    for cond in conditions:
        sub = df[df["condition"] == cond]
        rews = sub["sealed_mean_reward"].tolist()
        succs = sub["sealed_success_rate"].tolist()
        perrs = sub["sealed_pred_err"].tolist()
        summary_rows.append({
            "condition": cond,
            "n_replicates": len(sub),
            "mean_reward": float(np.mean(rews)),
            "ci95_reward": _ci95(rews),
            "mean_success": float(np.mean(succs)),
            "ci95_success": _ci95(succs),
            "mean_pred_err": float(np.mean(perrs)),
            "mean_env_steps": float(sub["env_steps_used"].mean()),
            "mean_wall_s": float(sub["wall_seconds"].mean()),
            "mean_energy": float(sub["energy_used"].mean()),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / f"{mode}_summary.csv", index=False)

    # Generate comparison plot from actual measurements
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y_pos = np.arange(len(sum_df))
    axes[0].barh(y_pos, sum_df["mean_reward"], xerr=sum_df["ci95_reward"], capsize=3, color="#2563eb", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(sum_df["condition"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Sealed Mean Reward (±95% CI)")
    axes[0].set_title(f"CORTEX-ECCNR ({mode}): Sealed Reward")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(y_pos, sum_df["mean_success"], xerr=sum_df["ci95_success"], capsize=3, color="#059669", alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sum_df["condition"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Sealed Success Rate (±95% CI)")
    axes[1].set_title(f"CORTEX-ECCNR ({mode}): Task Success Rate")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"{mode}_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    payload = {
        "mode": mode,
        "dataset_id": ref.dataset_id,
        "provenance_tier": ref.provenance_tier,
        "topology_hash": ref.topology_hash(),
        "n_nodes": ref.n_nodes,
        "n_edges": ref.n_edges,
        "total_wall_seconds": round(time.perf_counter() - t_start, 2),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "successive_action_experiment": succ_action,
        "summary": summary_rows,
    }
    (out_dir / f"{mode}_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CORTEX-ECCNR experiments")
    parser.add_argument("--mode", choices=["smoke", "pilot", "primary", "bptt_primary", "advance_verify"], default="smoke")
    parser.add_argument("--data-dir", type=Path, default=Path("data/flywire"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_suite(args.mode, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
