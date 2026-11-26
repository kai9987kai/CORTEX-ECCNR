# CORTEX-ECCNR Experiment Ledger

| Run ID | Mode | Dataset | Lineages × Gens | Replicates | Env Budget | Status |
|---|---|---|---:|---:|---:|---|
| `pytest_suite` | Unit & Integration | `synthetic_fixture` | — | 5 tests | — | **DONE** (5/5 passed in 5.37s) |
| `acquire_flywire` | Data Acquisition | FlyWire FAFB v783 | — | 1 | — | **DONE** (139,255 nodes, 15,091,983 edges, 54,492,922 synapses) |
| `smoke` | Smoke Test | `synthetic_fixture_48` | 2 × 2 | 1 | 800 | **DONE** (5 conditions passed) |
| `pilot` | Pilot | `flywire_fafb_v783_sub256` | 4 × 3 | 2 | 3,200 | **DONE** (20/20 runs completed) |
| `primary` | Primary Matrix (Eligibility) | `flywire_fafb_v783_sub256` | 4 × 3 | 5 | 4,800 | **DONE** (50/50 runs, 486.62s total) |
| `bptt_primary` | High-Budget BPTT Matrix | `flywire_fafb_v783_sub256` | 4 × 3 | 3 | 24,000 | **DONE** (30/30 runs, 2029.46s total) |
| `advance_verify` | v0.2.0 Gated-Router + Fisher/EWC Suite | `flywire_fafb_v783_sub256` | 4 × 3 | 3 | 12,000 | **DONE** (15/15 runs, 844.28s total) |
| `pytest_v02` | Expanded Unit/Integration/GUI Suite | `synthetic_fixture` | — | 11 tests | — | **DONE** (11/11 passed in 36.27s) |
| `successive_action` | Handoff vs Single | `flywire_fafb_v783_sub256` | 3 descendants | 1 | 600 | **INCLUDED** in pilot & primary |
| `antigravity_assisted` | Optional AI-guided | — | — | — | — | **OPTIONAL / NOT RUN** (no live Antigravity teacher budget spent; kept distinct from scripted rules) |
| `scale_1000` | Optional 1000-lineage | — | — | — | — | **OPTIONAL / NOT RUN** (excluded to preserve core CPU budget) |
