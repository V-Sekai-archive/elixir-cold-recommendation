# Proposal: Checkpoint layout

Sub-proposal of the [documentation index](README.md). How RecGPT checkpoints are structured and loaded in Elixir.

---

## Problem or limitation

RecGPT weights are distributed as PyTorch checkpoints (e.g. [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model)). Elixir needs a well-defined export format and loader contract so inference and training can use the same params without ad-hoc conversion.

---

## Proposed improvement

Define a **checkpoint layout**: components (GPT-2, FSQ embedding, aux encoder, head), how to obtain and export (manifest + .npy), and mapping from export keys to inference. All export and load paths are specified so implementers can add tooling without guessing.

---

## Components

| Component               | Purpose             | Expected keys / shapes                      |
| ----------------------- | ------------------- | ------------------------------------------- |
| **GPT-2 backbone**      | Causal transformer  | `gpt2model.*` (or equivalent).              |
| **FSQ token embedding** | (15361, 768) lookup | `wte` or GPT-2 `wte` resized to 15361 rows. |
| **Auxiliary encoder**   | 192→768, LayerNorm  | `ae.*` or `linear*`, `norm*`.               |
| **Prediction head**     | Linear(768, 15361)  | `pred_head.weight`, `pred_head.bias`.       |

---

## Obtaining a checkpoint

**Download:** `mix recgpt.fetch_ckpt` downloads the model to `data/recgpt_layer_3_weight.pt` (or `--out <path>`).

---

## Export (manifest + .npy)

The loader expects **zip-based** .pt (PyTorch 1.6+). If `data.pkl` inside the zip has a bad CRC (e.g. some Hugging Face or CDN downloads), the loader falls back to reading that entry without CRC check so export still succeeds. Shape mismatches in the pickle are corrected from the actual byte size so tensors load reliably.

To load in Elixir, export to a directory with `manifest.json` and one `.npy` file per tensor:

| Source           | Command                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| PyTorch `.pt`    | `mix recgpt.export_ckpt --from-pt path/to/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export` |
| Existing export  | `mix recgpt.export_ckpt --from-export data/recgpt_ckpt_export --out other_dir`                    |
| Params in memory | `RecGPT.CheckpointExport.write_export(params, "data/recgpt_ckpt_export")`                         |

Inspect `manifest.json` in the export dir for exact keys and shapes.

---

## Mapping to inference

| Export key    | Elixir use                                                                 |
| ------------- | -------------------------------------------------------------------------- |
| `wte`         | FSQ token embedding table; shape `{15361, 768}`; Nx.gather by token ids.   |
| `ae.*`        | Aux linear 192→768 + LayerNorm; applied to `Training.encode_aux/3` output. |
| `gpt2model.*` | GPT-2 blocks; naming may be mapped in the loader.                          |
| `pred_head.*` | Linear layer; logits (batch, 15361).                                       |

`RecGPT.CheckpointLoader.load_from_export/1` returns `%{key => Nx.Tensor}` for use with `RecGPT.Inference`.

---

## Sub-proposals

- **Components** (above) — GPT-2, FSQ embedding, aux encoder, head.
- **Export** (above) — manifest + .npy; `export_ckpt --from-pt` or `write_export/2`.
- **Mapping to inference** (above) — Export key → Elixir use.

---

## See also

- [04 RecGPT library](04_recgpt_library.md) — Module reference.
- [02 Pipeline overview](02_pipeline_overview.md) — Checkpoint in the pipeline.
- [09 Parity overview](09_parity_overview.md) — Checkpoint key compatibility.
