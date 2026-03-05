# Dependencies

- **Nx**, **EXLA**, **Axon** — Tensors, EXLA backend (XLA/CUDA), and training. Inference and serve use EXLA.
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Jcs**, **Npy** — JSON, RFC 8785 canonicalization (Polymarket embedding text), and `.npy` checkpoint files.
- **grpc** — gRPC server for `mix recgpt.serve`.
- **Req** — HTTP (e.g. fetch_steam).
