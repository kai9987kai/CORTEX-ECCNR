# CORTEX-ECCNR Research Protocol

## 1. Central Research Question
Can brains derived from a complete biological connectome (FlyWire FAFB v783) collectively acquire, transmit, and consolidate skills across evolutionary lineages and Developmental Trace Replay (DTR) better than a single continuously trained brain under **equivalent environment step budgets**?

## 2. Prior-Work Separation
- **CORTEX-INTERVENE**: Prior 4-condition Antigravity-supervision study (not present in local workspace or `kai9987kai` GitHub; not modified).
- **Sonapse-LAB** (`https://github.com/kai9987kai/Sonapse-LAB`, commit `363681d`, Apache-2.0): Inspected read-only for observation/action contracts (`155`-dim sensory vector, `5`-dim bounded motor vector `[drive, turn, reach, height, grip]`), Dale's law sign convention, homeostatic/neuromodulatory coupling, and FlyWire FAFB v783 asset definitions. Not modified.
- **CORTEX-ECCNR**: Standalone repository extending the programme with evolutionary lineages, parent-conditioned repair, ancestry-aware exact-inheritance fusion, pre-adaptation, and Developmental Trace Replay (DTR).

## 3. Resource & Compute Budgets
- **Hardware**: 8 logical CPU cores, 16.76 GB RAM, CPU-first NumPy/SciPy execution.
- **Pilot**: 4 lineages × 3 generations on a 256-node role-stratified FlyWire FAFB v783 subgraph, 2 independent replicates (`seeds=[101, 202]`), 3,200 matched environment steps per replicate.
- **Primary Experiment**: 4 lineages × 3 generations, 5 independent replicates (`seeds=[101, 202, 303, 404, 505]`), **4,800 matched environment steps** per replicate across all 10 conditions, evaluated on **5 sealed held-out seeds** (`[9001, 9002, 9003, 9004, 9005]`).
- **Statistical Unit**: Independent run/population replicate ($N = 5$); 95% confidence intervals computed at the run level, never inflated by correlated episode steps.

## 4. Staged Experimental Matrix (Matched Env Steps)
1. `fixed_connectome`: FlyWire substrate, frozen recurrent topology & weights, readout-only training.
2. `single_plastic`: Single brain with continuous weight + online structural plasticity.
3. `sequential_correction`: Single lineage evolved sequentially ($4 \times 3 = 12$ generations).
4. `parallel_best`: 4 parallel lineages × 3 generations, selecting the best individual.
5. `fusion_only`: `hybrid_exact` fusion + pre-adaptation, no replay.
6. `fusion_ordinary_replay`: `hybrid_exact` fusion + `sensory_action` replay.
7. `fusion_activity_replay`: `hybrid_exact` fusion + `sensory_activity` DTR.
8. `full_dtr_finetune`: `hybrid_exact` fusion + `full_dtr` + autonomous fine-tuning.
9. `ann_analogue`: Conventional dense `RecurrentANNAnalogue` (256 hidden units).
10. `rewired_control`: Degree-preserving rewired FlyWire control with `full_dtr_finetune`.
11. **Successive-Action Population Study**: 3 descendants handing off explicit state (`rates`, `neuromod`) vs 1 recurrent brain with matched total training steps.
