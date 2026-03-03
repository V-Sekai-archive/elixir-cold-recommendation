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
config :recgpt, :ckpt_expected_sha256,
  "b93219448d9800cf1c1b86ab265dfa5ccc6b29aef11c0795b1d376fb7971c82b"

# Inference dtype: {:f, 32}, {:bf, 16} for BF16, or :f8_e4m3fn (default) for FP8 (EXLA 0.11, Tensor Cores).

config :recgpt, :inference_dtype, :f8_e4m3fn

# Ablation: fix beam width (e.g. 1 for greedy). RECGPT_BEAM_WIDTH_OVERRIDE=1 to test.
config :recgpt, :beam_width_override,
  (case System.get_env("RECGPT_BEAM_WIDTH_OVERRIDE") do
     nil -> nil
     s -> case Integer.parse(s) do
            {n, _} when n >= 1 -> n
            _ -> nil
          end
   end)

# SLO: RecGPT latency targets (combination system; reflex-logic-market + bs-p add <0.1 ms).
# Primary target P50 = 20 ms; P99 budget from E2E ceiling. Override via RECGPT_TARGET_P50_MS / RECGPT_TARGET_P99_MS.
config :recgpt, :target_p50_ms,
  (System.get_env("RECGPT_TARGET_P50_MS") || "20") |> String.to_integer()

config :recgpt, :target_p99_ms,
  (System.get_env("RECGPT_TARGET_P99_MS") || "60") |> String.to_integer()

# Padded KV cache length for incremental forward. Shape (batch, n_head, max_cache_len, head_dim).
config :recgpt, :max_cache_len, 128

# Context cache: warm at load with step-0 results for these contexts (list of item_id lists).
# Enables cache hit and skips step 0 in recommend when context matches. Default [] = no warming.
# Example: [[] , [0]] warms empty context and context [0]. Use context_cache_warm_batch_size for batching.
config :recgpt, :context_cache_warm_list, []
config :recgpt, :context_cache_warm_batch_size, 4

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
