# Dependencies

- **Nx**, **EXLA**, **Axon** — Tensors, EXLA backend/compiler (XLA), and training. Inference and serve use EXLA.
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Npy** — JSON and `.npy` checkpoint files.
- **grpc** — gRPC server for `mix recgpt.serve`.
- **Req** — HTTP (e.g. fetch_ckpt, fetch_steam).
