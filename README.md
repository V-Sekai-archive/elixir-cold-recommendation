# RecGPT (Elixir)

Elixir library for **RecGPT-style sequential recommendation**: data pipeline, gRPC serving, Ecto/SQLite. **RecGPT inference, eval, and Predict run in Elixir** (RecGPT.Serve, RecGPT.Eval, fixture + checkpoint); no Python at runtime.

**RecGPT** ([paper](https://arxiv.org/abs/2506.06270), [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)) treats items as 4-token sequences; this repo provides the data pipeline (fetch, build_fixture), gRPC API, and Elixir-native inference and evaluation.

---

## Quick start

1. Get a checkpoint: `mix recgpt.fetch_ckpt` then `mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export`
2. Generate data and eval: `mix recgpt.first_step` (or `mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture && mix recgpt.eval`)
3. Serve: `RECGPT_FIXTURE=data/steam/fixture.json RECGPT_CKPT_EXPORT=data/recgpt_ckpt_export mix recgpt.serve`

→ Full steps and options: [docs/51_quick_start.md](docs/51_quick_start.md), [docs/02_pipeline_overview.md](docs/02_pipeline_overview.md), [docs/03_pipeline_steps.md](docs/03_pipeline_steps.md).

---

## Documentation

| Topic | Doc |
|-------|-----|
| **Quick start** | [docs/51_quick_start.md](docs/51_quick_start.md) |
| **Pipeline** | [docs/52_pipeline_summary.md](docs/52_pipeline_summary.md) |
| **Mix tasks** | [docs/53_mix_tasks.md](docs/53_mix_tasks.md) |
| **Modules** | [docs/54_modules_overview.md](docs/54_modules_overview.md) |
| **Dependencies** | [docs/55_dependencies.md](docs/55_dependencies.md) |
| **Dev container (Torchx)** | [docs/56_dev_container.md](docs/56_dev_container.md) |
| **Tests** | [docs/57_tests.md](docs/57_tests.md) |
| **gRPC API** | [docs/58_grpc_serve.md](docs/58_grpc_serve.md), [docs/01_grpc_api.md](docs/01_grpc_api.md) |
| **Versioning & references** | [docs/59_versioning_and_references.md](docs/59_versioning_and_references.md) |

**Full index:** [docs/README.md](docs/README.md) — library reference, pipeline, eval, checkpoint layout, parity, and more.
