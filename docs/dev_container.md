# Dev container (Torchx)

Inference and serve run on **Torchx** (LibTorch). The dev container provides a Linux environment; for **Windows**, use the host (Torchx supports Windows with bundled LibTorch).

1. Open the project in VS Code/Cursor and run **Reopen in Container** (or use the `.devcontainer/devcontainer.json` image with your tooling).
2. The container forwards ports 50051 and 50052; `postCreateCommand` runs `mix deps.get` and installs grpcurl.
3. Inside the container: `mix test`, `mix recgpt.serve` (with `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT` set).

**Windows:** Run `mix deps.get` and `mix compile` on the host. Torchx ships with LibTorch; no CUDA env vars are required for CPU. For GPU on Windows, ensure CUDA drivers match the Torchx/CUDA build if you use a CUDA-enabled Torchx variant.

**Latency:** On CPU, one Predict can be ~1–2s. With CUDA (when available), latency is typically lower after the first JIT.
