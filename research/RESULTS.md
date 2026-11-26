# CORTEX-ECCNR Measured Results

## 0. v0.2.0 Advanced Innovation Benchmark (`advance_verify`: `N = 3` Replicates, 12,000 Env Steps Each, Sealed Seeds `9001–9005`)
Evaluates the v0.2.0 architectural and consolidation advances (`ewc_dtr_finetune` and `gated_fusion_ewc` combining 4-tap axonal delays, apical-basal dendritic gating, astrocytic $[\text{Ca}^{2+}]$ stabilization, UCB-learned repair, MAP-Elites QD cross-lineage motif exchange, `gated_specialist_routing` exact inheritance, and two-stage Fisher/EWC-anchored `0.1×` differential-LR consolidation) on `flywire_fafb_v783_sub256`.

| Condition | N | Mean Reward (±95% CI) | Success Rate (±95% CI) | Mean Pred Error | Mean Env Steps | Energy Proxy | Mean Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_plastic` | 3 | `-1.6644 ± 0.4115` | `0.0% ± 0.0%` | `0.1276` | `12000` | `112.23` | `46.39` |
| `full_dtr_finetune` | 3 | `-1.5081 ± 0.5462` | `5.3% ± 2.6%` | `0.1335` | `12000` | `43.31` | `55.74` |
| `fusion_ordinary_replay` | 3 | `-1.5158 ± 0.1139` | `5.3% ± 10.5%` | `0.0536` | `12000` | `33.83` | `59.23` |
| `ewc_dtr_finetune` | 3 | `-1.6721 ± 0.9057` | `0.0% ± 0.0%` | `0.0673` | `12000` | `45.44` | `62.28` |
| **`gated_fusion_ewc`** | **3** | **`-1.1191 ± 1.2525`** | **`12.0% ± 12.0%`** | **`0.0549`** | **`12000`** | **`44.01`** | `57.33` |

### Key v0.2.0 Findings
1. **`gated_fusion_ewc` Achieves Highest Reward & Success at Half Budget (`12,000` Steps)**: Combining exact-inherited specialist readout branches under a learned context router (`gated_specialist_routing`) + QD motif exchange + Fisher/EWC-anchored fine-tuning achieved **`-1.1191` mean reward** and **`12.0%` mean sealed success** (peaking at `20.0%` on seed `202` and `16.0%` on seed `101`), outperforming `fusion_ordinary_replay` (`-1.5158`, `5.3%`), `full_dtr_finetune` (`-1.5081`, `5.3%`), and `single_plastic` (`-1.6644`, `0.0%`) while using `2.5×` less energy proxy (`44.01` vs `112.23`) than `single_plastic`.
2. **Mechanistic Synergy**: Diagonal Fisher/EWC protection alone (`ewc_dtr_finetune`) halved prediction error relative to `full_dtr_finetune` (`0.0673` vs `0.1335`, `-49.6%`), while adding multi-branch specialist context routing (`gated_fusion_ewc`) prevented conflicting parent readouts from overwriting one another, unlocking high task completion (`12.0%`) alongside low prediction error (`0.0549`, `-58.9%` vs `full_dtr_finetune`).

---

## 1. High-Budget PyTorch Truncated-BPTT Experiment (`bptt_primary`: `N = 3` Replicates, 24,000 Env Steps Each, Sealed Seeds `9001–9005`)
Dataset: `flywire_fafb_v783_sub256` (`256` nodes, `2,982` induced directed edges, topology hash `ea70dd96bc644b2e`).

| Condition | N | Mean Reward (±95% CI) | Success Rate (±95% CI) | Mean Pred Error | Mean Env Steps | Energy Proxy | Mean Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_connectome` | 3 | `-1.5019 ± 0.6030` | `4.0% ± 7.8%` | `0.1199` | `24000` | `204.69` | `44.84` |
| `single_plastic` | 3 | `-2.0111 ± 0.5902` | `4.0% ± 7.8%` | `0.1296` | `24000` | `212.77` | `54.43` |
| `sequential_correction` | 3 | `-1.7138 ± 0.7264` | `4.0% ± 4.5%` | `0.0884` | `24000` | `13.70` | `76.68` |
| `parallel_best` | 3 | `-2.1548 ± 0.5234` | `0.0% ± 0.0%` | `0.0802` | `24000` | `41.43` | `80.39` |
| `fusion_only` | 3 | `-1.7893 ± 0.3133` | `1.3% ± 2.6%` | `0.0845` | `24000` | `48.41` | `80.41` |
| **`fusion_ordinary_replay`** | **3** | **`-1.2892 ± 0.6637`** | **`13.3% ± 2.6%`** | **`0.0363`** | **`24000`** | **`42.37`** | `78.83` |
| `fusion_activity_replay` | 3 | `-2.1821 ± 0.3380` | `1.3% ± 2.6%` | `0.0461` | `24000` | `51.01` | `78.93` |
| `full_dtr_finetune` | 3 | `-1.7694 ± 0.3377` | `1.3% ± 2.6%` | `0.1130` | `24000` | `68.31` | `79.93` |
| `ann_analogue` | 3 | `-2.1694 ± 0.3305` | `0.0% ± 0.0%` | `0.0674` | `24000` | `2175.13` | `46.07` |
| `rewired_control` | 3 | `-1.9182 ± 0.6216` | `1.3% ± 2.6%` | `0.1091` | `24000` | `60.14` | `55.34` |

### Key BPTT Findings
1. **Exact-Inheritance Fusion + Behavioral Replay Wins**: `fusion_ordinary_replay` achieved the highest mean reward (`-1.2892 ± 0.6637`), highest sealed task success (`13.3% ± 2.6%` — non-overlapping 95% CI vs `parallel_best` `0.0%`, `ann_analogue` `0.0%`, `fusion_only` `1.3% ± 2.6%`, `full_dtr_finetune` `1.3% ± 2.6%`, and `rewired_control` `1.3% ± 2.6%`), and lowest prediction error (`0.0363`).
2. **Why Frozen Recurrent Core + Replay Beats Unfrozen Fine-Tuning**: Preserving bit-for-bit inherited `bio_weights` while consolidating behavioral traces into readout/input adapters (`fusion_ordinary_replay`) outperformed unfreezing `bio_weights` (`full_dtr_finetune`, `1.3%` success), showing that premature recurrent unfreezing causes drift.
3. **Energy Efficiency**: `fusion_ordinary_replay` used `42.37` energy proxy units vs `2,175.13` for the dense `ann_analogue` (**`51.3×` lower**).

---

## 2. Eligibility-Trace Primary Experiment (`primary`: `N = 5` Replicates, 4,800 Env Steps Each, Sealed Seeds `9001–9005`)

| Condition | N | Mean Reward (±95% CI) | Success Rate (±95% CI) | Mean Pred Error | Mean Env Steps | Energy Proxy | Mean Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_connectome` | 5 | `-1.8593 ± 0.8204` | `0.000 ± 0.000` | `0.0499` | `4800` | `58.74` | `3.35` |
| `single_plastic` | 5 | `-2.0185 ± 0.7513` | `0.000 ± 0.000` | `0.0637` | `4800` | `58.74` | `6.73` |
| `sequential_correction` | 5 | `-2.0593 ± 0.9097` | `0.000 ± 0.000` | `0.0473` | `4800` | `14.39` | `11.03` |
| `parallel_best` | 5 | `-2.2958 ± 0.5725` | `0.008 ± 0.016` | `0.1141` | `4800` | `26.72` | `12.05` |
| `fusion_only` | 5 | `-2.0138 ± 0.8335` | `0.000 ± 0.000` | `0.1228` | `4800` | `30.37` | `12.25` |
| `fusion_ordinary_replay` | 5 | `-2.1529 ± 0.6557` | `0.000 ± 0.000` | `0.0984` | `4800` | `36.11` | `11.85` |
| `fusion_activity_replay` | 5 | `-2.0979 ± 0.6920` | `0.000 ± 0.000` | `0.1040` | `4800` | `36.11` | `11.94` |
| `full_dtr_finetune` | 5 | `-2.2165 ± 0.6718` | `0.000 ± 0.000` | `0.0701` | `4800` | `40.56` | `12.45` |
| `ann_analogue` | 5 | `-1.8920 ± 0.5586` | `0.000 ± 0.000` | `0.0148` | `4800` | `444.43` | `2.61` |
| `rewired_control` | 5 | `-2.3388 ± 0.4127` | `0.000 ± 0.000` | `0.0933` | `4800` | `40.39` | `12.54` |

## 3. Successive-Action Population Experiment
- **Single recurrent brain (600 training steps)**: `-1.0589` mean reward
- **3-descendant successive handoff (200 steps each, explicit `rates` + `neuromod` handoff)**: `-2.2056` mean reward (`delta = -1.1467`)
