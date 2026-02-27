# RecGPT: use Torchx as the default Nx backend (CPU/GPU tensor ops).
# Requires :torchx in deps and a successful mix compile.
import Config

config :nx, default_backend: Torchx.Backend

# UCI Clickstream SQLite repo. Override with RECGPT_DATABASE_PATH.
config :recgpt, RecGPT.Repo, []
