# RecGPT (Elixir)

Elixir library for **RecGPT-style sequential recommendation**: data pipeline, gRPC serving, Ecto/SQLite. **RecGPT inference, eval, and Predict run in Elixir** (RecGPT.Serve, RecGPT.Eval, fixture + checkpoint); no Python at runtime.

**RecGPT** ([paper](https://arxiv.org/abs/2506.06270), [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)) treats items as 4-token sequences; this repo provides the data pipeline (fetch, build_fixture), gRPC API, and Elixir-native inference and evaluation.

---

## Quick start

1. **Get a checkpoint** (export dir with `manifest.json` + `.npy` tensors):

   ```bash
   mix recgpt.fetch_ckpt
   mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
   ```

2. **Generate data and run first step** (Steam baseline, Elixir eval):

   ```bash
   mix recgpt.first_step                     # fetch → build_fixture → eval (Elixir)
   # Or: mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture && mix recgpt.eval
   ```

3. **Serve recommendations** (gRPC; Predict uses Elixir Serve):

   ```bash
   RECGPT_FIXTURE=data/steam/fixture.json RECGPT_CKPT_EXPORT=data/recgpt_ckpt_export mix recgpt.serve
   ```

See [Pipeline](#pipeline), [docs/02_pipeline_overview.md](docs/02_pipeline_overview.md), and [docs/03_pipeline_steps.md](docs/03_pipeline_steps.md) for the full sequence and options.

---

## Pipeline

| Step | Command / API | Outputs |
|------|----------------|---------|
| **1. Data** | `mix recgpt.fetch_steam data/steam` or `RecGPT.Steam.Fetch.run/1` | `items.json`, `train_sequences.json`, `test_sequences.json`, `cold_test_sequences.json`, `cold_train_sequences.json` |
| **2. Fixture** | `mix recgpt.build_fixture` or `RecGPT.FixtureBuild.build/2` | `fixture.json` (`num_items`, `token_id_list`) |
| **3. Pretrain** | `mix recgpt.pretrain` (uses `AxonTrain.stream_batches` + `run/3`) | Updated checkpoint in `--out` |
| **4. Eval** | `mix recgpt.eval` (Elixir; `--data-dir`, `--ckpt`, `--fixture`, `--test`) | Hit@k, MRR, etc. |

For best quality, **pretrain then eval**; zero-shot (pretrained ckpt only) is a baseline. See [docs/07_steam_splits_and_pretraining.md](docs/07_steam_splits_and_pretraining.md).

**Canonical item texts (same input as official):** By default, `build_fixture` and `compare_embeddings` read item text from the `canonical_item_texts` SQLite table so both use the same bytes. To populate that table from Python (byte-exact match with the official script), run once:

```bash
mix ecto.migrate
uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --verify
```

Use the same `--db` or `RECGPT_SQLITE_PATH` as Elixir. Then both inputs are from Python; no Python at runtime.

---

## Mix tasks

| Task | Purpose |
|------|---------|
| `mix recgpt.fetch_ckpt` | Download RecGPT PyTorch checkpoint from Hugging Face (hkuds/RecGPT_model). |
| `mix recgpt.export_ckpt` | Export checkpoint to `manifest.json` + `.npy` (from `--from-pt`). |
| `mix recgpt.fetch_steam` | Fetch Steam test split from HuggingFace (hkuds/RecGPT_dataset); write items + train/test/cold sequences. |
| `mix recgpt.build_fixture` | Build `fixture.json` from `items.json` (Embedding + FSQ). Options: `--items`, `--out`, `--ckpt`. |
| `mix recgpt.pretrain` | Pretrain on `train_sequences.json` with fixture + checkpoint; write updated params to `--out`. |
| `mix recgpt.eval` | Run next-item eval in Elixir (`--data-dir`, `--ckpt`, `--fixture`, `--test`). |
| `mix recgpt.serve` | Start gRPC server (port 50051): Predict uses RecGPT.RecommendationService (default: Serve). Set `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`. |

Paths default to `data/steam/` and `thirdparty/checkpoints/recgpt`. Env: `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT` for serve.

**Catalog storage** uses object-store semantics; options are BEAM-native (file path, [CubDB](https://hex.pm/packages/cubdb), or RabbitMQ's [Khepri](https://hex.pm/packages/khepri)). See [docs/13_infrastructure_serving.md](docs/13_infrastructure_serving.md#catalog-storage-object-store-semantics).

---

## Modules (overview)

| Module | Purpose |
|--------|---------|
| **RecGPT.FSQ** | FSQ quantizer (levels [8,8,8,6,5], 4 tokens/item, vocab 15360). `load_params/1`, `encode/2`. |
| **RecGPT.FSQEncoder** | Embeddings (num_items, 768) + FSQ params → `token_id_list` (list of 4-token lists). |
| **RecGPT.Embedding** | Text → 768-d via Bumblebee (all-mpnet-base-v2). `encode_item_text_dict/1`. |
| **RecGPT.FixtureBuild** | Build fixture from items.json. `build/2`, `write_fixture/2`. |
| **RecGPT.Training** | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2`. |
| **RecGPT.AxonTrain** | Training loop: `stream_batches/4`, `run/3` (Polaris optimizer). |
| **RecGPT.Inference** | Forward pass (training): token embed + aux + GPT-2 + head. `forward/4`, `forward_full_sequence/4`. |
| **RecGPT.InferenceParams** | Build defn-friendly full params (atom keys). Stub checkpoints get identity layers so one code path. |
| **RecGPT.InferenceDefn** | Defn entry points for serve: `forward_with_cache/4`, `forward_incremental/5` (EXLA JIT). |
| **RecGPT.Serve** | Load state (fixture + checkpoint); EXLA JIT only. Implements `RecGPT.RecommendationService`. |
| **RecGPT.CheckpointLoader** | Load export dir → `%{key => Nx.Tensor}`. |
| **RecGPT.CheckpointExport** | Write params to export dir (manifest + .npy). |
| **RecGPT.Steam.Fetch** | Steam test split → items + train/test/cold sequences (HuggingFace hkuds/RecGPT_dataset). |

Full list and details: [docs/04_recgpt_library.md](docs/04_recgpt_library.md).

---

## Dependencies

- **Nx**, **EXLA**, **Axon** — Tensors, EXLA backend/compiler, and training. Inference and serve use EXLA only.
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Npy** — JSON and `.npy` checkpoint files.
- **grpc** — gRPC server for `mix recgpt.serve`.
- **Req** — HTTP (e.g. fetch_ckpt, fetch_steam).

---

## Dev container (EXLA)

Inference and serve run on **EXLA** only (no Torchx). Use the dev container for a supported EXLA environment:

1. Open the project in VS Code/Cursor and run **Reopen in Container** (or use the `.devcontainer/devcontainer.json` image with your tooling).
2. The container forwards ports 50051 and 50052; `postCreateCommand` runs `mix deps.get`.
3. Inside the container: `mix test`, `mix recgpt.serve` (with `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT` set).

**GPU (CUDA):** The dev container requests GPU access (`--gpus all`) and sets `EXLA_TARGET=cuda12` so EXLA can use the GPU. On the host, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so Docker can expose the GPU. If you have no GPU or want CPU-only, set `EXLA_TARGET=cpu` and `XLA_TARGET=cpu` in your environment (or in `remoteEnv` in devcontainer.json).

EXLA compiles the inference graph on first use; the first request may be slower.

---

## Tests

```bash
mix test --no-start
```

- Excluded by default: `integration`, `eval`.
- **Include integration:** `mix test --include integration`
- **Eval (fixture + ckpt + test file):** `mix test test/recgpt/eval_test.exs --include eval --include integration`

See [docs/06_evaluation_and_testing.md](docs/06_evaluation_and_testing.md) and [docs/04_recgpt_library.md](docs/04_recgpt_library.md).

---

## gRPC API (serve)

Service is gRPC-only. Contract: [priv/proto/recgpt/v1/recommendation.proto](priv/proto/recgpt/v1/recommendation.proto). [docs/01_grpc_api.md](docs/01_grpc_api.md).

- **recgpt.v1.PredictionService/Predict** — Request: `context_item_ids`, `max_results`; response: `item_ids`, `items` (ItemSummary).

Errors use gRPC status (e.g. INVALID_ARGUMENT, UNAVAILABLE). See [recommendation.proto](priv/proto/recgpt/v1/recommendation.proto).

---

## Documentation

| Doc | Content |
|-----|---------|
| [docs/README.md](docs/README.md) | **Documentation index** — all docs by topic and quick reference. |
| [docs/04_recgpt_library.md](docs/04_recgpt_library.md) | Full module reference, deps, tests. |
| [docs/09_parity_overview.md](docs/09_parity_overview.md) | Python RecGPT parity: task list, validation. |
| [docs/08_recgpt_checkpoint_layout.md](docs/08_recgpt_checkpoint_layout.md) | Checkpoint state_dict, export, loader. |
| [docs/06_evaluation_and_testing.md](docs/06_evaluation_and_testing.md) | Zero-shot vs trained, null hypothesis, held-out eval. |
| [docs/05_eval_data_shapes.md](docs/05_eval_data_shapes.md) | JSON shapes: test_sequences, items, fixture, train_sequences, cold. |
| [docs/07_steam_splits_and_pretraining.md](docs/07_steam_splits_and_pretraining.md) | Train/test/cold splits, pretrain-first pipeline. |
| [docs/02_pipeline_overview.md](docs/02_pipeline_overview.md), [docs/03_pipeline_steps.md](docs/03_pipeline_steps.md) | Pipeline overview and steps: commands, options, file layout. |
| [docs/22_top_tier_recommendations.md](docs/22_top_tier_recommendations.md) | Top-tier improvements: typespecs, Dialyzer, integration test, health, benchmarks. |
| [priv/proto/recgpt/v1/recommendation.proto](priv/proto/recgpt/v1/recommendation.proto) | gRPC API contract (PredictionService.Predict). |

---

## Versioning

See [CHANGELOG.md](CHANGELOG.md). Bump the version in `mix.exs` and tag releases (e.g. `v0.2.0`) for meaningful releases.

---

## References

- [RecGPT paper](https://arxiv.org/abs/2506.06270)
- [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)
- [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model)
- [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) (Steam splits)
