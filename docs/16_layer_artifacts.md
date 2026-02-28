# Layer 1: Artifacts

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Next: [17 Layer Representation](17_layer_representation.md).

---

## What it does

Read and write external artifacts: HuggingFace/Steam JSON, PyTorch .pt (zip), and the export directory (manifest.json + .npy). No RecGPT-specific logic beyond file layout.

## Public surface

Steam.Fetch.run/1, RecGPT.PtLoader, RecGPT.CheckpointLoader.load_from_export/1, RecGPT.CheckpointExport.write_export/2.

## How to test

Unit tests with temporary directories and fixture files; no dependency on other RecGPT layers.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
- [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md).