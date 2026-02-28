# Layer 1: Artifacts

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Next: [17 Layer Representation](17_layer_representation.md).

---

## Problem or limitation

External artifacts (Steam JSON, .pt, export dir) must be read and written in a defined way; without a single documented surface, layout and loader contracts are unclear.

---

## Proposed improvement

Document Layer 1 (Artifacts): responsibility, public surface, and how to test. No RecGPT business logic beyond file layout.

Read and write external artifacts: HuggingFace/Steam JSON, PyTorch .pt (zip), and the export directory (manifest.json + .npy). **PtLoader** loads zip-based .pt and, when `data.pkl` has a bad CRC, falls back to reading that entry raw; pickle shapes are normalized to the actual tensor size. **Public surface:** Steam.Fetch.run/1, RecGPT.PtLoader, RecGPT.CheckpointLoader.load_from_export/1, RecGPT.CheckpointExport.write_export/2. **How to test:** Unit tests with temporary directories and fixture files; no dependency on other RecGPT layers.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
- [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md).
