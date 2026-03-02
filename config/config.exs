# RecGPT: EXLA backend for Nx (CUDA in devcontainer). All Nx/Defn run on EXLA.
# For low latency: default_client :cuda. With :host (CPU), Predict ~1–2s; with :cuda ~300–400ms after setup (COG, 12-layer).
# Devcontainer: .devcontainer/ has Dockerfile + devcontainer.json (EXLA/CUDA 12.9).
import Config

config :nx, default_backend: EXLA.Backend
config :nx, :default_defn_options, compiler: EXLA
config :exla, :default_client, :cuda

# Request batching for Predict: collect up to predict_batch_size requests or wait predict_batch_timeout_ms.
# Default 1 and 0 = one request at a time (no batching).
config :recgpt, :predict_batch_size, 1
config :recgpt, :predict_batch_timeout_ms, 0

# Checkpoint integrity: SHA256 must match. Compute with: mix recgpt.ckpt_sha256 --ckpt data/recgpt_ckpt_export
# Or set RECGPT_CKPT_SHA256 env var.
config :recgpt, :ckpt_expected_sha256, "b93219448d9800cf1c1b86ab265dfa5ccc6b29aef11c0795b1d376fb7971c82b"

# EXLA JIT disk cache: persist compiled XLA binaries for faster setup on restart.
# Set RECGPT_EXLA_CACHE_DIR to override path; empty string disables caching.
config :recgpt, :exla_jit_cache_dir, System.get_env("RECGPT_EXLA_CACHE_DIR") || "tmp/exla_cache"

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]

# Waffle: blob/artifact storage (local by default; set WAFFLE_ASSET_HOST or use S3 in prod).
config :waffle,
  storage: Waffle.Storage.Local,
  asset_host: System.get_env("WAFFLE_ASSET_HOST") || "http://localhost:4000"
