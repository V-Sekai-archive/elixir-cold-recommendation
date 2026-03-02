# RecGPT: Torchx backend. Nx ops run on LibTorch (CPU or CUDA on Windows/Linux).
# Defn uses Nx.Defn.Evaluator so defn code runs on the default backend (Torchx.Backend).
# Torchx does not implement Nx.Defn.Compiler (__jit__/5); do not set compiler: Torchx.
import Config

config :nx, default_backend: Torchx.Backend

# Request batching for Predict: collect up to predict_batch_size requests or wait predict_batch_timeout_ms.
# Default 1 and 0 = one request at a time (no batching).
config :recgpt, :predict_batch_size, 1
config :recgpt, :predict_batch_timeout_ms, 0

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]

# Waffle: blob/artifact storage (local by default; set WAFFLE_ASSET_HOST or use S3 in prod).
config :waffle,
  storage: Waffle.Storage.Local,
  asset_host: System.get_env("WAFFLE_ASSET_HOST") || "http://localhost:4000"
