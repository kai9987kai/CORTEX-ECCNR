# CORTEX-ECCNR Data & Code Provenance

## 1. Biological Connectome Dataset
- **Dataset**: FlyWire FAFB v783 Adult Female Drosophila Whole-Brain Connectome
- **Version**: `v783` (connections & root IDs) + `flywire_annotations v2.1.0`
- **License**: `CC-BY-NC-4.0` (FAFB v783); `CC-BY-4.0` (flywire_annotations v2.1.0)
- **Citations**:
  - Dorkenwald et al. (2024). *Neuronal wiring diagram of an adult brain*. Nature 634, 124–138. DOI: `10.1038/s41586-024-07558-y`
  - Schlegel et al. (2024). *Whole-brain annotation and multi-connectome cell typing of Drosophila*. Nature 634, 139–152. DOI: `10.1038/s41586-024-07686-5`
- **Verified Full-Brain Statistics** (from `data/flywire/cache/manifest.json`):
  - **Nodes**: `139,255` proofread neurons
  - **Raw neuropil-split contact rows**: `16,847,997`
  - **Aggregated unique directed edges (`pre -> post`)**: `15,091,983`
  - **Total anatomical synapses**: `54,492,922` (multiplicity min `1`, median `2.0`, max `2,405`)
  - **Annotation coverage**: `139,255 / 139,255` (`100.0%`); neurotransmitter-annotated: `138,654` (`99.57%`)
  - **Role breakdown**: `sensory: 16,903`, `visual: 86,113`, `central: 32,468`, `descending: 1,303`, `ascending: 2,362`, `motor: 106`
- **Source File Checksums**:
  - `proofread_connections_783.feather`: MD5 `f48f972d262323a102aed49af1396b8a`, SHA-256 `24f960ae3e7d4f8cd30db3b62e99fb5179cc3d1e76d8c155bfb441e9737d3faf`
  - `proofread_root_ids_783.npy`: MD5 `e0e6c19732fd8c7a4e39a2d170105421`, SHA-256 `7c7b7e818e9232e5ab64793d52ba20574dbe9da3a6509fbc74193b0f259a01be`
  - `annotations_v2.1.0.tsv`: MD5 `8527e0a95ed5f112766b13260a91e8e2`, SHA-256 `30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6`

## 2. Modelling Assumptions
- Anatomical synapse counts $c_{ij}$ are converted to initial simulated weights via $w_{ij} = s_i \log(1 + c_{ij})$ with incoming row-sum cap $0.90$, where $s_i \in \{+1, -1, 0\}$ follows Dale's law (`ACh: +1`, `GABA/Glut/Hist: -1`, monoamines/unknown: `0`).
- This is a **graph-derived rate connectome model**, not a physiological conductance-based fly brain.

## 3. Reference Repository Inspection
- `Sonapse-LAB` (`https://github.com/kai9987kai/Sonapse-LAB`, commit `363681d`, Apache-2.0): inspected read-only; no files modified.
