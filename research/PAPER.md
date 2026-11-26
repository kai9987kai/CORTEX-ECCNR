# CORTEX-ECCNR: Evolutionary Connectome Consolidation and Neural Replay

## Abstract
We investigate whether neural agents derived from a verified adult *Drosophila* whole-brain connectome (**FlyWire FAFB v783**) can collectively acquire, repair, fuse, and consolidate sensorimotor skills across evolutionary lineages better than a single continuously trained brain under strictly matched environment-step budgets. We build a standalone, reproducible CPU-first research platform and interactive observatory (`cortex-eccnr` v0.2.0) featuring copy-on-write (CoW) sparse connectome weights, Dale's law sign preservation, axonal conduction delays, two-compartment dendritic gating, astrocytic $[\text{Ca}^{2+}]$ stabilization, UCB-learned repair with a MAP-Elites Quality-Diversity archive, 7-strategy ancestry-aware graph fusion, and Fisher/EWC-anchored Developmental Trace Replay (DTR). Across a 5-replicate, 4,800-step eligibility-trace study, multi-lineage ECCNR did not surpass baselines due to step-budget dilution. When scaled to **24,000 matched steps with PyTorch truncated-BPTT** (`N = 3` replicates, sealed seeds `9001–9005`), **exact-inheritance fusion + behavioral replay (`fusion_ordinary_replay`) achieved `-1.2892 ± 0.6637` reward, `13.3% ± 2.6%` sealed success (vs `0.0%` for `parallel_best` and `ann_analogue`), `0.0363` prediction error, and `51.3×` lower energy proxy (`42.37` vs `2,175.13`) than the dense ANN analogue.** Furthermore, in our v0.2.0 `12,000`-step verification suite (`advance_verify`), **`gated_fusion_ewc` (multi-branch specialist context routing + QD motif exchange + Fisher/EWC `0.1×` differential-LR unfreezing) achieved `-1.1191 ± 1.2525` sealed reward and `12.0%` mean success (up to `20.0%`), outperforming `fusion_ordinary_replay` (`-1.5158`, `5.3%`), `full_dtr_finetune` (`-1.5081`, `5.3%`), and `single_plastic` (`-1.6644`, `0.0%`) at half the step budget while cutting `full_dtr_finetune` prediction error by `58.9%`.**

## 1. Related Work & Separation from Prior Studies
- **Drosophila Connectomics**: FlyWire FAFB v783 (Dorkenwald et al., 2024, *Nature*; Schlegel et al., 2024, *Nature*).
- **Sonapse-LAB** (`kai9987kai/Sonapse-LAB`, commit `363681d`, Apache-2.0): Provides the 155-dim sensory / 5-dim motor interface and Dale's law sign conventions.
- **Separation**: This study is explicitly separate from the prior 4-condition `CORTEX-INTERVENE` study and does not modify `Sonapse-LAB`.

## 2. Hypotheses
- **H1**: Evolutionary lineage repair + exact-inheritance fusion + DTR outperforms a single plastic brain under equal environment-step budgets. *(Refuted at 4,800 steps)*
- **H2**: Ancestry-aware exact-inheritance fusion (`hybrid_exact`) outperforms selecting the best single parent (`parallel_best`). *(Supported directionally: `-2.0138` vs `-2.2958`, overlapping CIs)*
- **H3**: Full DTR improves predictive calibration over fusion without replay. *(Supported: pred error `0.0701` vs `0.1228`)*
- **H4**: Biological FlyWire topology outperforms a degree-preserving rewired control. *(Supported directionally: `-2.2165` vs `-2.3388`, overlapping CIs)*

## 3. Dataset & Architecture Details
- **Full Connectome**: FlyWire FAFB v783 (`139,255` nodes, `15,091,983` aggregated directed edges, `54,492,922` synapses, `100%` node annotation coverage).
- **Runtime Substrate**: Role-stratified `256`-node induced subgraph (`2,982` directed edges, topology hash `ea70dd96bc644b2e`), CoW `float32` weights, sparse structural overlays, and `ECCNRWorld` (`navigation`, `dynamic_obstacles`, `memory_cue`, `resource_allocation`, `sequential`).

## 4. Controls, Metrics, and Statistical Analysis
Ten conditions (`fixed_connectome`, `single_plastic`, `sequential_correction`, `parallel_best`, `fusion_only`, `fusion_ordinary_replay`, `fusion_activity_replay`, `full_dtr_finetune`, `ann_analogue`, `rewired_control`) were evaluated across `N = 5` independent replicates (`seeds = [101, 202, 303, 404, 505]`) on `5` sealed held-out seeds (`[9001..9005]`). Run-level means and 95% confidence intervals ($\pm 1.96 \cdot s / \sqrt{N}$) are reported in `research/RESULTS.md` and plotted in `results/primary_comparison.png`.

## 5. Measured Results & Negative Results
- **Primary Negative Result**: Under a strict `4,800`-step total budget, dividing experience across `4` lineages × `3` generations (`~270–320` steps per lineage-generation) prevented lineages from reaching task-completion regimes (`0.0%` sealed success across most conditions, `0.8%` for `parallel_best`), whereas `fixed_connectome` (`-1.8593 ± 0.8204`) and `ann_analogue` (`-1.8920 ± 0.5586`) achieved slightly higher mean reward.
- **Successive-Action Negative Result**: Handing off state across 3 descendants (`200` steps each) yielded `-2.2056` vs `-1.0589` for a single recurrent brain (`600` steps).
- **Positive Mechanistic Signals**: Fusion improved over `parallel_best` (`+0.2820`), DTR cut prediction error by `42.9%`, biological topology beat rewired controls (`+0.1223`), and sparse FlyWire substrates used `7.6×–30.9×` less energy proxy than the dense ANN.

## 6. Limitations, Threats to Validity, and Reproducibility
See `research/LIMITATIONS.md` for full limitations (256-node subgraph vs 139k full brain, eligibility-trace learning vs BPTT, deferred 1,000-lineage and live Antigravity conditions). All code, configs, tests (`5/5` passing), and scripts are in `cortex-eccnr/`.

## 7. AI Assistance Disclosure
This codebase, experimental harness, and report were implemented and executed with Antigravity AI assistance. All reported metrics come directly from executed runs (`pytest`, `acquire_flywire.py`, `run_experiments.py`) with zero fabricated measurements.
