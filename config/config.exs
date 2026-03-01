# RecGPT: use Torchx as the default Nx backend (CPU/GPU tensor ops).
# Requires :torchx in deps and a successful mix compile.
import Config

config :nx, default_backend: Torchx.Backend

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]
