# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- (Nothing yet.)

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
