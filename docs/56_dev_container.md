# Dev container (EXLA)

Inference and serve run on **EXLA** (XLA). The dev container provides a Linux environment with CUDA 12 for GPU; for **Windows**, use the host (EXLA supports Windows with the host client or CUDA if built for it).

## Local deps with Homebrew

To run Elixir and Docker locally (e.g. to use the dev container or run `mix` on the host):

```bash
# Docker (required for dev container; on macOS use --cask)
brew install docker          # Linux
brew install --cask docker   # macOS (Docker Desktop)

# Elixir + Erlang (for mix deps.get, mix compile, mix test on host)
brew install elixir
```

Then start Docker (macOS: open Docker Desktop; Linux: `sudo systemctl start docker`), open the repo in Cursor/VS Code, and use **Dev Containers: Reopen in Container**. Or run Mix on the host:

```bash
mix local.hex --force && mix local.rebar --force
mix deps.get
mix compile
```

1. Open the project in VS Code/Cursor and run **Reopen in Container** (or use the `.devcontainer/devcontainer.json` image with your tooling).
2. The container forwards ports 50051 and 50052; `postCreateCommand` runs `mix deps.get` and installs grpcurl and EXLA libs.
3. Inside the container: `mix test`, `mix recgpt.serve` (with `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT` set).

**Windows:** Run `mix deps.get` and `mix compile` on the host. EXLA uses the host (CPU) client by default. For GPU on Windows, set `EXLA_TARGET` and ensure CUDA drivers match the EXLA build.

**Latency:** On CPU (host client), one Predict can be ~1–2s. With CUDA (RTX 4090, 12-layer): setup ~20–30s, predict ~300–400ms (Replicate COG: setup = one-time compile, predict = per-request).
