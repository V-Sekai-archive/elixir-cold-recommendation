# RecGPT (Elixir)

Elixir library for **RecGPT-style sequential recommendation**: data pipeline, gRPC serving, Ecto/SQLite. **RecGPT inference, eval, and Predict run in Elixir** (RecGPT.Serve, RecGPT.Eval, fixture + checkpoint); no Python at runtime.

**RecGPT** ([paper](https://arxiv.org/abs/2506.06270), [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)) treats items as 4-token sequences; this repo provides the data pipeline (fetch, build_fixture), gRPC API, and Elixir-native inference and evaluation.

---

## Quick start

1. Get checkpoint and data: `mix recgpt.refetch` (FuXi-Linear init + VAE + Steam).
2. Eval: `mix recgpt.first_step` (or `mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture && mix recgpt.eval`)
3. Serve: `RECGPT_FIXTURE=data/steam/fixture.json RECGPT_CKPT_EXPORT=data/fuxi_ckpt_export mix recgpt.serve`

→ Full steps and options: [docs/features/51_quick_start.md](docs/features/51_quick_start.md), [docs/features/02_pipeline_overview.md](docs/features/02_pipeline_overview.md), [docs/features/03_pipeline_steps.md](docs/features/03_pipeline_steps.md).

---

## Documentation

| Topic | Doc |
|-------|-----|
| **Quick start** | [docs/features/51_quick_start.md](docs/features/51_quick_start.md) |
| **Pipeline** | [docs/features/02_pipeline_overview.md](docs/features/02_pipeline_overview.md), [docs/features/03_pipeline_steps.md](docs/features/03_pipeline_steps.md) |
| **Mix tasks** | [docs/features/53_mix_tasks.md](docs/features/53_mix_tasks.md) |
| **Modules** | [docs/features/54_modules_overview.md](docs/features/54_modules_overview.md) |
| **Dependencies** | [docs/features/55_dependencies.md](docs/features/55_dependencies.md) |
| **Dev container (Torchx)** | [docs/features/56_dev_container.md](docs/features/56_dev_container.md) |
| **Tests** | [docs/features/57_tests.md](docs/features/57_tests.md) |
| **gRPC API** | [docs/features/58_grpc_serve.md](docs/features/58_grpc_serve.md), [docs/features/01_grpc_api.md](docs/features/01_grpc_api.md) |
| **Versioning & references** | [docs/features/59_versioning_and_references.md](docs/features/59_versioning_and_references.md) |
| **Formal model (Lean)** | [formal/README.md](formal/README.md) — FSQ codec + trie decode certified via `plausible-witness-dag` |

**Full index:** [docs/README.md](docs/README.md) — library reference, pipeline, eval, checkpoint layout, parity, and more.
