# Checkpoints (thirdparty)

Checkpoint artifacts for RecGPT live here so they stay under `thirdparty/` and are not mixed with app `data/`. Binary files (`.pt`, `.npy`) are gitignored; populate via the Mix tasks below.

## Layout

| Path | Contents | How to populate |
|------|----------|-----------------|
| **vae/** | VAE checkpoint: `vae_len4_fsq88865_ep90.pt` | `mix recgpt.fetch_vae_ckpt` (default `--out` is this dir) |
| **recgpt/** | RecGPT model: Elixir export (`manifest.json` + `*.npy`) and optionally the source `.pt` | `mix recgpt.fetch_ckpt` then `mix recgpt.export_ckpt --from-pt ... --out thirdparty/checkpoints/recgpt` |

## Commands

```bash
# VAE (for FSQ tokenizer parity; used by build_fixture --vae-ckpt and by optional Python pipeline)
mix recgpt.fetch_vae_ckpt
# Writes thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt

# RecGPT model: download .pt then export to Elixir format
mix recgpt.fetch_ckpt
mix recgpt.export_ckpt --from-pt thirdparty/checkpoints/recgpt/recgpt_layer_3_weight.pt --out thirdparty/checkpoints/recgpt
# recgpt/ then contains manifest.json, *.npy, and optionally the .pt
```

Use `--out` / `--ckpt` to override paths. Defaults for eval and serve can point at `thirdparty/checkpoints/recgpt` when using this layout.
