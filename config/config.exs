# RecGPT: EXLA backend for Nx (CUDA in devcontainer). All Nx/Defn run on EXLA.
# For low latency: default_client :cuda. With :host (CPU), Predict ~1-2s; with :cuda ~300-400ms after setup.
import Config

config :nx, default_backend: EXLA.Backend
config :nx, :default_defn_options, compiler: EXLA
config :exla, :default_client, :cuda

# Request batching for Predict: collect up to predict_batch_size requests or wait predict_batch_timeout_ms.
config :recgpt, :predict_batch_size, 1
config :recgpt, :predict_batch_timeout_ms, 0

# Max time (ms) for a single Predict recommend; avoids indefinite hang on first-request JIT. Default 120s.
config :recgpt, :predict_timeout_ms, 120_000

# Checkpoint integrity: SHA256 must match when set. Default nil (FuXi has no fixed SHA; pretrained varies).
config :recgpt,
       :ckpt_expected_sha256,
       System.get_env("RECGPT_CKPT_SHA256") || nil

# Inference dtype: {:f, 32} or {:bf, 16} for BF16 (Tensor Cores).
config :recgpt, :inference_dtype, {:bf, 16}

# Decode strategy: :beam_search (default) or :mtp (Multi-Token Prediction).
config :recgpt,
       :decode_strategy,
       (case System.get_env("RECGPT_DECODE_STRATEGY", "beam_search") do
          "mtp" -> :mtp
          "lookahead" -> :mtp
          "direct_score" -> :mtp
          _ -> :beam_search
        end)

# SLO: RecGPT latency targets.
config :recgpt,
       :target_p50_ms,
       (System.get_env("RECGPT_TARGET_P50_MS") || "20") |> String.to_integer()

config :recgpt,
       :target_p99_ms,
       (System.get_env("RECGPT_TARGET_P99_MS") || "60") |> String.to_integer()

config :recgpt, ecto_repos: []

# Load env-specific config (config/dev.exs, config/test.exs, etc.)
import_config "#{config_env()}.exs"
