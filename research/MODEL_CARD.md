# CORTEX-ECCNR Model & Substrate Capability Card

- **Model Family**: Sparse FlyWire-Derived Recurrent Rate Connectome (`ConnectomeBrain` v0.2.0)
- **Founding Connectome**: FlyWire FAFB v783 (`139,255` proofread adult female *Drosophila* neurons, `15,091,983` directed edges, `54,492,922` synapses); runtime induced subgraph `flywire_fafb_v783_sub256` (`256` role-stratified neurons, `2,982` directed edges, topology hash `ea70dd96bc644b2e`).
- **Observation / Action Interface**: `155`-dim sensory vector (`45×3` retina + `8` proprioception + `6` task one-hot + `4` cue/stamina + `2` reward/time) -> `5`-dim bounded motor vector (`[drive, turn, reach, height, grip]`, compatible with `Sonapse-LAB`).
- **Biophysical Abstractions**: Dale's law sign preservation (`ACh +1`, `GABA/Glut/Hist -1`, monoamines `0`), 4-tap axonal delay buffer, apical-basal dendritic predictive gating, astrocytic $[\text{Ca}^{2+}]$ stabilization, adaptive homeostatic thresholds, and sparse developmental neurogenesis/synaptogenesis/pruning with persistent innovation IDs.
- **Consolidation Architecture**: Ancestry-aware exact-inheritance fusion (`hybrid_exact` and `gated_specialist_routing`) + prioritized Developmental Trace Replay (DTR) + diagonal Fisher/EWC-anchored `0.1×` differential recurrent LR truncated-BPTT fine-tuning.
- **Intended Scope & Boundaries**: Graph-derived computational neuroscience & evolutionary consolidation research model; **not** a complete physiological fly brain and not intended for safety-critical physical control without separate validation.
