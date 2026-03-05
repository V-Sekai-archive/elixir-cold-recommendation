# Checkpoints (thirdparty)

Checkpoint artifacts for RecGPT live here so they stay under `thirdparty/` and are not mixed with app `data/`. Binary files (`.pt`, `.npy`) are gitignored; populate via the Mix tasks below.

## Layout

| Path | Contents | How to populate |
|------|----------|-----------------|
| **vae/** | VAE checkpoint: `vae_len4_fsq88865_ep90.pt` | `mix recgpt.fetch_vae_ckpt` (default `--out` is this dir) |

## Commands

```bash
# VAE (for FSQ tokenizer parity; used by build_fixture --vae-ckpt and by optional Python pipeline)
mix recgpt.fetch_vae_ckpt
# Writes thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt
```

Use `mix recgpt.refetch` for the full pipeline (FuXi checkpoint to `data/fuxi_ckpt_export`, VAE, Steam). Use `--out` / `--ckpt` to override paths.
