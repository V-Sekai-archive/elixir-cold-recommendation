# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **EXLA JIT disk cache:** Persist compiled XLA binaries to `tmp/exla_cache` (or `RECGPT_EXLA_CACHE_DIR`); restarts skip recompilation. Cache key includes EXLA/Nx versions, checkpoint SHA256, and dtype.
- **BF16 inference:** Config `inference_dtype: {:bf, 16}` for Tensor Core acceleration; default remains FP32.
- **Padded KV cache:** Fixed shape `(batch, n_head, max_cache_len, head_dim)` for stable EXLA JIT cache keys; `max_cache_len` (default 128) in config.
- **Adaptive beam width:** Beam width `max(4, min(top_k + 2, 12))` to reduce wasted work for small `top_k`.
- **Checkpoint SHA256 validation:** `mix recgpt.ckpt_sha256`; config `ckpt_expected_sha256` for integrity checks.
- **trace_predict:** `--runs` and `--jitter-ms` for statistical significance when profiling.
- **Devcontainer:** `.devcontainer/` with EXLA/CUDA 12.9 for reproducible builds.
- **Performance plans:** `docs/plans/` with plan status, estimates, and implementation order.

### Changed

- **Decode:** SPMD-only; list-based decode removed.
- **Config:** `exla_jit_cache_dir`, `inference_dtype`, `max_cache_len` in `config.exs`.
- **CONTRIBUTING:** Format, Credo `--strict`, Dialyzer in code quality checklist.

### Fixed

- **Code quality:** Credo (length/1, negated conditions, Enum.map_join, matches in `if`); Dialyzer ignores for Stream typing; `.dialyzer_ignore.exs` and `.credo.exs` updates.

## [0.1.0] - 2026-02-27

### Added

- **API:** gRPC `recgpt.v1.PredictionService` with `Predict` RPC; contract in `priv/proto/recgpt/v1/recommendation.proto`.
- **Pipeline:** Mix tasks `fetch_ckpt`, `export_ckpt`, `fetch_steam`, `build_fixture`, `pretrain`, `eval`, `serve`; single reproducible path from data to trained model and metrics.
- **Core modules:** `RecGPT.FSQ`, `RecGPT.FSQEncoder`, `RecGPT.Embedding` (MPNet/Bumblebee), `RecGPT.FixtureBuild`, `RecGPT.Training`, `RecGPT.AxonTrain`, `RecGPT.Inference`, `RecGPT.Decode`, `RecGPT.Trie`, `RecGPT.Serve`, `RecGPT.Eval`; checkpoint `RecGPT.CheckpointLoader`, `RecGPT.CheckpointExport`, `RecGPT.PtLoader`; data `RecGPT.Steam.Fetch`.
- **Checkpoint:** Export/load via manifest + `.npy`; optional PyTorch `.pt` import.
- **Eval:** Hit@k, MRR, null baseline comparison; test/cold splits; `RecGPT.Eval.evaluate/3`, `load_test_cases/1`.
- **Serving:** `mix recgpt.serve` (gRPC port 50051); HTTP health server on port 50052 for readiness; `RecGPT.ReleaseTasks.serve/0` for releases; Dockerfile.
- **Quality:** Typespecs (`@spec`) on public APIs; Dialyzer in CI; Credo; property-based tests (StreamData); Benchee script `bench/recgpt_serve_bench.exs`.
- **Catalog:** `RecGPT.Catalog.write!/2` for SSD-stable atomic writes; optional `--catalog` path at serve time.
- **Docs:** Pipeline, gRPC API, eval data shapes, layers and testing, infrastructure and release, top-tier recommendations.
