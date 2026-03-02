# Dev container (EXLA)

Inference and serve run on **EXLA** (XLA). The dev container provides a Linux environment with CUDA 12 for GPU; for **Windows**, use the host (EXLA supports Windows with the host client or CUDA if built for it).

1. Open the project in VS Code/Cursor and run **Reopen in Container** (or use the `.devcontainer/devcontainer.json` image with your tooling).
2. The container forwards ports 50051 and 50052; `postCreateCommand` runs `mix deps.get` and installs grpcurl and EXLA libs.
3. Inside the container: `mix test`, `mix recgpt.serve` (with `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT` set).

**Windows:** Run `mix deps.get` and `mix compile` on the host. EXLA uses the host (CPU) client by default. For GPU on Windows, set `EXLA_TARGET` and ensure CUDA drivers match the EXLA build.

**Latency:** On CPU (host client), one Predict can be ~1–2s. With CUDA (when available), latency is typically lower after the first JIT.
