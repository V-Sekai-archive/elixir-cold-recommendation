# RecGPT: use Torchx with CUDA so tensors and training use GPU RAM.
# Build Torchx with LIBTORCH_TARGET (e.g. cu129) for CUDA. Fitting in 24 GB: --batch-size 4 or 8, cap catalog (e.g. --limit 100).
import Config

config :nx, default_backend: {Torchx.Backend, device: :cuda}

# Request batching for Predict: collect up to predict_batch_size requests or wait predict_batch_timeout_ms.
# Default 1 and 0 = one request at a time (no batching).
config :recgpt, :predict_batch_size, 1
config :recgpt, :predict_batch_timeout_ms, 0

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]
