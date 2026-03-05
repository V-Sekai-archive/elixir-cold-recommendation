# Third-party / checkpoints

This directory holds **checkpoint artifacts** (VAE and RecGPT model binaries) and optional **datasets** so this project can run inference and eval without depending on external servers. No vendored Python code; RecGPT inference, eval, and gRPC Predict run entirely in Elixir.

## KuaiRand-Pure (Phase 1 pretraining)

Place the dataset in any directory (e.g. `thirdparty/KuaiRand-Pure` or `C:\Users\<user>\Desktop\KuaiRand-Pure` on Windows). Then:

```bash
mix recgpt.convert_trajectories --from /path/to/KuaiRand-Pure --out data/kuairand --format kuairand
```

Requires `log_standard_*.csv`, `log_random_*.csv`, and optionally `video_features_basic_pure.csv`. Download from https://kuairand.com/. See [docs/features/93_pretraining_plan.md](../docs/features/93_pretraining_plan.md).

## checkpoints/

Checkpoint binaries live under **checkpoints/** and are not committed (see [checkpoints/README.md](checkpoints/README.md)):

- **vae/** — VAE `.pt`; default for `mix recgpt.fetch_vae_ckpt`.

Use `data/fuxi_ckpt_export` (from `mix recgpt.refetch`) as `--ckpt` for eval/serve.
