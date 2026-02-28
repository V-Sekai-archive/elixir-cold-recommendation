# RecGPT: use Torchx as the default Nx backend (CPU/GPU tensor ops).
# Requires :torchx in deps and a successful mix compile.
import Config

config :nx, default_backend: Torchx.Backend
