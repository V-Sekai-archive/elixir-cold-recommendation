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

# Max time (ms) for a single Predict recommend; avoids indefinite hang on first-request JIT. Default 120s.
config :recgpt, :predict_timeout_ms, 120_000

# Checkpoint integrity: SHA256 must match. Compute with: mix recgpt.ckpt_sha256 --ckpt data/recgpt_ckpt_export
# Or set RECGPT_CKPT_SHA256 env var. Overridden to nil in config/test.exs for stub exports.
config :recgpt,
       :ckpt_expected_sha256,
       "b93219448d9800cf1c1b86ab265dfa5ccc6b29aef11c0795b1d376fb7971c82b"

# Inference dtype: {:f, 32} or {:bf, 16} for BF16 (Tensor Cores).

config :recgpt, :inference_dtype, {:bf, 16}

# SLO: RecGPT latency targets (combination system; reflex-logic-market + bs-p add <0.1 ms).
# Primary target P50 = 20 ms; P99 budget from E2E ceiling. Override via RECGPT_TARGET_P50_MS / RECGPT_TARGET_P99_MS.
config :recgpt,
       :target_p50_ms,
       (System.get_env("RECGPT_TARGET_P50_MS") || "20") |> String.to_integer()

config :recgpt,
       :target_p99_ms,
       (System.get_env("RECGPT_TARGET_P99_MS") || "60") |> String.to_integer()

# SQLite catalog/token storage (optional). Set RECGPT_SQLITE_PATH to use. Run mix ecto.migrate.
config :recgpt, ecto_repos: [RecGPT.Repo]

config :recgpt, RecGPT.Repo,
  database: System.get_env("RECGPT_SQLITE_PATH") || "priv/recgpt.sqlite3",
  migration_primary_key: [type: :integer]

# Waffle: blob/artifact storage (local by default; set WAFFLE_ASSET_HOST or use S3 in prod).
config :waffle,
  storage: Waffle.Storage.Local,
  asset_host: System.get_env("WAFFLE_ASSET_HOST") || "http://localhost:4000"

# Load env-specific config (config/dev.exs, config/test.exs, etc.)
import_config "#{config_env()}.exs"
