# Dev container (EXLA)

Inference and serve run on **EXLA** only (no Torchx). Use the dev container for a supported EXLA environment:

1. Open the project in VS Code/Cursor and run **Reopen in Container** (or use the `.devcontainer/devcontainer.json` image with your tooling).
2. The container forwards ports 50051 and 50052; `postCreateCommand` runs `mix deps.get`.
3. Inside the container: `mix test`, `mix recgpt.serve` (with `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT` set).

**XLA target (EXLA_TARGET / XLA_TARGET):** Exact versioning and requirements are in [XLA (hexdocs.pm)](https://hexdocs.pm/xla/0.10.0/XLA.html). Summary:

| Value   | Target / requirements |
|--------|------------------------|
| `cpu`  | Host CPU only (default; no nvcc/CUDA needed). |
| `cuda12` | CUDA ≥ 12.1, cuDNN ≥ 9.8 and < 10.0, NCCL ≥ 2.27, NVSHMEM ≥ 3.3. Use precompiled binary; check version with `nvcc --version`. |
| `cuda13` | CUDA ≥ 13.0, cuDNN ≥ 9.12 and < 10.0, NCCL ≥ 2.27, NVSHMEM ≥ 3.3. |

The dev container is configured for **CUDA 12.9** and **XLA build from source** (`XLA_BUILD=true`, `XLA_TARGET=cuda12`). Using CUDA 12 ensures the container runtime matches typical host drivers (e.g. driver 576.x supports CUDA 12.9; CUDA 13 requires r580+). The XLA binary is built locally so it matches the image's CUDA/cuDNN/NVSHMEM versions; the **first** `mix deps.get` or `mix compile` can take a long time (see [XLA — Building from source](https://hexdocs.pm/xla/0.10.0/XLA.html#building-from-source)). Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host and run with `--gpus all` (already in `runArgs`).

EXLA compiles the inference graph on first use; the first request may be slower.

**Latency:** With EXLA on CPU (`:host`), one Predict can be ~1–2s. For ~100ms-class latency, run on GPU: the dev config sets `default_client` to `:cuda` when available. Ensure the container has GPU access and CUDA libs; then the first request pays JIT cost, later ones are fast. To force CPU in dev (e.g. no GPU), set `RECGPT_EXLA_HOST=1`.

**If you see** `nvshmem_transport_ibrc.so.3: cannot open shared object file`: run once as root in the container: `./scripts/setup_exla_libs.sh`. It creates the compat libs and updates the loader config. New containers run this automatically via `postCreateCommand`.
