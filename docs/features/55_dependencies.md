# Dependencies

- **Nx**, **EXLA**, **Axon** — Tensors, EXLA backend (XLA/CUDA), and training. Inference and serve use EXLA. See [79 Why EXLA over Torchx](79_exla_over_torchx.md).
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Jcs**, **Npy** — JSON, RFC 8785 canonicalization (Polymarket embedding text), and `.npy` checkpoint files.
- **grpc** — gRPC server for `mix recgpt.serve`.
- **Req** — HTTP (e.g. fetch_ckpt, fetch_steam).
