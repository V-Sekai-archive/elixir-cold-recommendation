# Dependencies

- **Nx**, **Torchx**, **Axon** — Tensors, Torchx backend (LibTorch), and training. Inference and serve use Torchx.
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Npy** — JSON and `.npy` checkpoint files.
- **grpc** — gRPC server for `mix recgpt.serve`.
- **Req** — HTTP (e.g. fetch_ckpt, fetch_steam).
