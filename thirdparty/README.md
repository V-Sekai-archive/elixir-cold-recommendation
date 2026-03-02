# Third-party / checkpoints

This directory holds **checkpoint artifacts** (VAE and RecGPT model binaries) so this project can run inference and eval without depending on external servers. No vendored Python code; RecGPT inference, eval, and gRPC Predict run entirely in Elixir.

## checkpoints/

Checkpoint binaries live under **checkpoints/** and are not committed (see [checkpoints/README.md](checkpoints/README.md)):

- **vae/** — VAE `.pt`; default for `mix recgpt.fetch_vae_ckpt`.
- **recgpt/** — RecGPT `.pt` and Elixir export (manifest + .npy); default for `mix recgpt.fetch_ckpt` and `mix recgpt.export_ckpt --out`.

Use `thirdparty/checkpoints/recgpt` as `--ckpt` for eval/serve when using this layout.
