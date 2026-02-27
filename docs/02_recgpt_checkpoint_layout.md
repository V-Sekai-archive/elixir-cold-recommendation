# RecGPT checkpoint state_dict layout

For loading [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) (or `recgpt_layer_3_weight.pt`) into the Elixir inference model.

## Expected components (from port estimate)

| Component | Purpose | Expected keys / shapes |
|-----------|---------|------------------------|
| **GPT-2 backbone** | Causal transformer | `gpt2model.*` (or similar) — match Bumblebee GPT-2 param names where possible |
| **FSQ token embedding** | (15361, 768) lookup | `wte` or equivalent; may be GPT-2 `wte` resized to 15361 |
| **Auxiliary encoder** | 192→768, LayerNorm | `ae.*` or `linear*`, `norm*` |
| **Prediction head** | Linear(768, 15361) | `pred_head.weight`, `pred_head.bias` |

## Checkpoint location

**Download from Hugging Face:** `mix recgpt.fetch_ckpt` downloads [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model/tree/main) to `data/recgpt_layer_3_weight.pt` (or use `--out` / `--base`).

The script looks for the checkpoint in this order:

1. `RECGPT_CKPT` environment variable
2. `thirdparty/RecGPT_repo/ckpt/recgpt_layer_3_weight.pt`
3. `thirdparty/RecGPT_model/recgpt_layer_3_weight.pt`

Or pass explicitly: `--ckpt path/to/recgpt_layer_3_weight.pt`.

## Discovering actual keys

After exporting (see above), open `manifest.json` in the export dir to see state_dict keys and shapes.

To export tensors for the Elixir loader (manifest.json + .npy):

- **From .pt (requires Python + torch):** `mix recgpt.export_ckpt --from-pt path/to/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export`
- **From existing export (pure Elixir):** `mix recgpt.export_ckpt --from-export data/recgpt_ckpt_export --out other_dir`
- **From params in memory:** `RecGPT.CheckpointExport.write_export(params, "data/recgpt_ckpt_export")`

Then load in Elixir via `RecGPT.CheckpointLoader.load_from_export/1` with the export directory path.

## Mapping to Elixir inference model

- **wte** → FSQ token embedding table; shape `{15361, 768}`; Nx.gather by token ids.
- **ae.*** → Aux linear 192→768 + LayerNorm; applied to `Training.encode_aux/3` output.
- **gpt2model.*** → Bumblebee GPT-2 :base; if names differ, use a one-off mapping in the loader.
- **pred_head** → Linear layer; output logits (batch, 15361).
