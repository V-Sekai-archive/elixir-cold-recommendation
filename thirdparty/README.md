# Third-party / checkpoints

This directory holds **checkpoint artifacts** (VAE and RecGPT model binaries) and **git submodules** so this project can run inference and eval without depending on external servers. No vendored Python code; RecGPT inference, eval, and gRPC Predict run entirely in Elixir.

## prediction-market-analysis (submodule)

Jon-Becker Polymarket dataset for `mix recgpt.convert_trajectories --format jon_becker`.

```bash
git submodule update --init thirdparty/prediction-market-analysis
cd thirdparty/prediction-market-analysis && make setup   # ~36 GiB
```

Then convert with:

```bash
mix recgpt.convert_trajectories --from thirdparty/prediction-market-analysis --out data/polymarket --format jon_becker
```

Convert uses Polymarket Gamma API for stable canonical JSON-LD item embedding text (RFC 8785 JCS). See [docs/92_polymarket_semantic_source.md](../docs/features/92_polymarket_semantic_source.md).

## checkpoints/

Checkpoint binaries live under **checkpoints/** and are not committed (see [checkpoints/README.md](checkpoints/README.md)):

- **vae/** — VAE `.pt`; default for `mix recgpt.fetch_vae_ckpt`.
- **recgpt/** — RecGPT `.pt` and Elixir export (manifest + .npy); default for `mix recgpt.fetch_ckpt` and `mix recgpt.export_ckpt --out`.

Use `thirdparty/checkpoints/recgpt` as `--ckpt` for eval/serve when using this layout.
