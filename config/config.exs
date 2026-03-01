# RecGPT: use Torchx with CUDA so tensors and training use GPU RAM.
# Build Torchx with LIBTORCH_TARGET (e.g. cu129) for CUDA. Fitting in 24 GB: --batch-size 4 or 8, cap catalog (e.g. --limit 100).
import Config

config :nx, default_backend: {Torchx.Backend, device: :cuda}

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]
