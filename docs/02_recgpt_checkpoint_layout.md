# Checkpoint layout

How RecGPT checkpoints (e.g. [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) or a local `recgpt_layer_3_weight.pt`) are structured and loaded in Elixir.

---

## Components

| Component | Purpose | Expected keys / shapes |
|-----------|---------|------------------------|
| **GPT-2 backbone** | Causal transformer | `gpt2model.*` (or equivalent). |
| **FSQ token embedding** | (15361, 768) lookup | `wte` or GPT-2 `wte` resized to 15361 rows. |
| **Auxiliary encoder** | 192→768, LayerNorm | `ae.*` or `linear*`, `norm*`. |
| **Prediction head** | Linear(768, 15361) | `pred_head.weight`, `pred_head.bias`. |

---

## Obtaining a checkpoint

**Download:** `mix recgpt.fetch_ckpt` downloads the model to `data/recgpt_layer_3_weight.pt` (or use `--out` / `--base`).

**Resolve path:** The task checks, in order: env `RECGPT_CKPT`; `thirdparty/RecGPT_repo/ckpt/recgpt_layer_3_weight.pt`; `thirdparty/RecGPT_model/recgpt_layer_3_weight.pt`. Or pass the path explicitly.

---

## Export (manifest + .npy)

To load in Elixir, export to a directory with `manifest.json` and one `.npy` file per tensor:

| Source | Command |
|--------|---------|
| PyTorch `.pt` | `mix recgpt.export_ckpt --from-pt path/to/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export` |
| Existing export | `mix recgpt.export_ckpt --from-export data/recgpt_ckpt_export --out other_dir` |
| Params in memory | `RecGPT.CheckpointExport.write_export(params, "data/recgpt_ckpt_export")` |

Inspect `manifest.json` in the export dir for exact keys and shapes.

---

## Mapping to inference

| Export key | Elixir use |
|------------|------------|
| `wte` | FSQ token embedding table; shape `{15361, 768}`; Nx.gather by token ids. |
| `ae.*` | Aux linear 192→768 + LayerNorm; applied to `Training.encode_aux/3` output. |
| `gpt2model.*` | GPT-2 blocks; naming may be mapped in the loader. |
| `pred_head.*` | Linear layer; logits (batch, 15361). |

`RecGPT.CheckpointLoader.load_from_export/1` returns `%{key => Nx.Tensor}` for use with `RecGPT.Inference`.

---

## See also

- [00 RecGPT library](00_recgpt_library.md) — Module reference.
- [08 Pipeline reference](08_pipeline_reference.md) — Checkpoint setup in the pipeline.
- [01 Python parity progress](01_python_recgpt_parity_progress.md) — Checkpoint key compatibility.
