# RecGPT (Elixir)

Elixir library for **RecGPT-style sequential recommendation**: FSQ (Finite Scalar Quantization), text embeddings (MPNet via Bumblebee), training pipeline, and HTTP serving. No GenServer; use from any application (e.g. polymarket).

**RecGPT** ([paper](https://arxiv.org/abs/2506.06270), [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)) treats items as 4-token sequences; this library provides the data pipeline, training (Axon + Polaris), inference, and eval tooling to match that setup.

---

## Quick start

1. **Get a checkpoint** (export dir with `manifest.json` + `.npy` tensors):

   ```bash
   mix recgpt.fetch_ckpt
   mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
   ```

2. **Generate data and build the pipeline** (UCI Clickstream example):

   ```bash
   mix recgpt.clickstream                    # items, train/test/cold sequences
   mix recgpt.build_fixture                  # items → fixture.json (Embedding + FSQ)
   mix recgpt.pretrain --out data/ckpt_out   # train on train_sequences, write updated checkpoint
   mix recgpt.eval                           # eval on test + cold_test (requires cold_test file)
   ```

3. **Serve recommendations** (optional):

   ```bash
   mix recgpt.serve --fixture data/clickstream/fixture.json --ckpt data/ckpt_out
   ```

See [Pipeline](#pipeline) and [docs/08_pipeline_reference.md](docs/08_pipeline_reference.md) for the full sequence and options.

---

## Pipeline

| Step | Command / API | Outputs |
|------|----------------|---------|
| **1. Data** | `mix recgpt.clickstream` or `RecGPT.Clickstream.Fetch.run/2` | `items.json`, `train_sequences.json`, `test_sequences.json`, `cold_test_sequences.json`, `cold_train_sequences.json` |
| **2. Fixture** | `mix recgpt.build_fixture` or `RecGPT.FixtureBuild.build/3` | `fixture.json` (`num_items`, `token_id_list`) |
| **3. Pretrain** | `mix recgpt.pretrain` (uses `AxonTrain.stream_batches` + `run/3`) | Updated checkpoint in `--out` |
| **4. Eval** | `mix recgpt.eval` (requires `--test` and `--cold-test` files) | Hit@k, MRR, cold-test metrics |

For best quality, **pretrain then eval**; zero-shot (pretrained ckpt only) is a baseline. See [docs/07_steam_splits_and_pretraining.md](docs/07_steam_splits_and_pretraining.md).

---

## Mix tasks

| Task | Purpose |
|------|---------|
| `mix recgpt.fetch_ckpt` | Download RecGPT PyTorch checkpoint from Hugging Face (hkuds/RecGPT_model). |
| `mix recgpt.export_ckpt` | Export checkpoint to `manifest.json` + `.npy` (from `--from-pt` or `--from-export`). |
| `mix recgpt.clickstream` | Fetch UCI Clickstream, run migrations, load SQLite; write items + train/test/cold sequences. |
| `mix recgpt.build_fixture` | Build `fixture.json` from `items.json` (Embedding + FSQ). Options: `--items`, `--out`, `--ckpt`, `--fsq`. |
| `mix recgpt.pretrain` | Pretrain on `train_sequences.json` with fixture + checkpoint; write updated params to `--out`. |
| `mix recgpt.eval` | Run next-item eval (Hit@k, MRR) on test + cold-test sets. Requires fixture, checkpoint, and both test files. |
| `mix recgpt.serve` | Start REST API: GET /v1/catalog/items, POST /v1/catalog:recommend, GET /v1/health. |

Paths default to `data/clickstream/` and `data/recgpt_ckpt_export`; override with `--fixture`, `--ckpt`, `--test`, `--cold-test`, etc. Env: `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`.

---

## Modules (overview)

| Module | Purpose |
|--------|---------|
| **RecGPT.FSQ** | FSQ quantizer (levels [8,8,8,6,5], 4 tokens/item, vocab 15360). `load_params/1`, `encode/2`. |
| **RecGPT.FSQEncoder** | Embeddings (num_items, 768) + FSQ params → `token_id_list` (list of 4-token lists). |
| **RecGPT.Embedding** | Text → 768-d via Bumblebee (all-mpnet-base-v2). `encode_item_text_dict/1`, `encode_texts/1`. |
| **RecGPT.FixtureBuild** | Build fixture from items or from precomputed embeddings. `build/3`, `build_from_embeddings/3`, `write_fixture/2`. |
| **RecGPT.Training** | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2`. |
| **RecGPT.AxonTrain** | Training loop: `stream_batches/4`, `run/3` (Polaris optimizer). |
| **RecGPT.Inference** | Forward pass: token embed + aux + GPT-2 + head. `forward/4`, `forward_full_sequence/4`. |
| **RecGPT.Serve** | Load state (fixture + checkpoint), `recommend/3`, `search/3`, item_ids_to_context_token_ids. |
| **RecGPT.Eval** | `evaluate/3`, `load_test_cases/1` (Hit@k, MRR, null rejection). |
| **RecGPT.Decode** | Beam search for next-item from logits + trie. |
| **RecGPT.CheckpointLoader** | Load export dir → `%{key => Nx.Tensor}`. |
| **RecGPT.CheckpointExport** | Write params to export dir (manifest + .npy). |
| **RecGPT.Clickstream.Fetch** | UCI Clickstream → SQLite + JSON artifacts; cold splits via `compute_cold_splits/4`. |

Full list and details: [docs/00_recgpt_library.md](docs/00_recgpt_library.md).

---

## Dependencies

- **Nx**, **Axon** — Tensors and training.
- **Bumblebee** (GitHub `main`) — MPNet text embeddings.
- **Jason**, **Npy** — JSON and `.npy` checkpoint files.
- **Plug.Cowboy** — HTTP server for `mix recgpt.serve`.
- **Ecto + SQLite** — Clickstream data.
- **Req** — HTTP (e.g. fetch_ckpt, Clickstream zip).
- **PropCheck** (dev/test) — Property-based tests.

---

## Tests

```bash
mix test --no-start
```

- Excluded by default: `embedding` (loads HF model), `integration`, `eval`, `e2e_serve`, `compare_python`, `pt_fixture`.
- **Include integration:** `mix test --include integration`
- **Include embedding:** `mix test --include embedding` (long timeout)
- **PropCheck:** `MIX_ENV=test mix run script/run_propcheck.exs`
- **Eval (fixture + ckpt + test file):** `mix test test/recgpt/eval_test.exs --include eval --include integration`

See [docs/05_evaluation_and_testing.md](docs/05_evaluation_and_testing.md) and [docs/00_recgpt_library.md](docs/00_recgpt_library.md).

---

## REST API (serve)

RESTful API following [Google API Design Guide](https://cloud.google.com/apis/design). Unified gRPC+REST: [docs/13](docs/13_grpc_rest_api.md), [docs/14](docs/14_api_schemas.md). REST: [docs/09](docs/09_rest_api.md). Only `/v1/` endpoints are served.

- **GET /v1/catalog/items?q=...&pageSize=20** — List (search) catalog items; response: `{"items": [{"item_id", "display_name"}]}`.
- **POST /v1/catalog:recommend** — Body: `{"context_item_ids": [0,1,2], "max_results": 5}` → `{"item_ids": [...], "items": [...]}`.
- **GET /v1/health** — Readiness: `{"status": "ok"}`.

Errors return a JSON body with `error.code`, `error.message`, `error.status`. See [docs/09_rest_api.md](docs/09_rest_api.md).

---

## Documentation

| Doc | Content |
|-----|---------|
| [docs/README.md](docs/README.md) | **Documentation index** — all docs by topic and quick reference. |
| [docs/00_recgpt_library.md](docs/00_recgpt_library.md) | Full module reference, deps, tests. |
| [docs/01_python_recgpt_parity_progress.md](docs/01_python_recgpt_parity_progress.md) | Python RecGPT parity: task list, validation, PropCheck. |
| [docs/02_recgpt_checkpoint_layout.md](docs/02_recgpt_checkpoint_layout.md) | Checkpoint state_dict, export, loader. |
| [docs/03_etnf_database_design.md](docs/03_etnf_database_design.md) | ETNF and database design steps. |
| [docs/04_foss_datasets_etnf_dublin_core_xmp.md](docs/04_foss_datasets_etnf_dublin_core_xmp.md) | Schema, Dublin Core, XMP JSON-LD (RDF/Grax). |
| [docs/05_evaluation_and_testing.md](docs/05_evaluation_and_testing.md) | Zero-shot vs trained, null hypothesis, held-out eval. |
| [docs/06_eval_data_shapes.md](docs/06_eval_data_shapes.md) | JSON shapes: test_sequences, items, fixture, train_sequences, cold. |
| [docs/07_steam_splits_and_pretraining.md](docs/07_steam_splits_and_pretraining.md) | Train/test/cold splits, pretrain-first pipeline. |
| [docs/08_pipeline_reference.md](docs/08_pipeline_reference.md) | End-to-end pipeline: commands, options, file layout. |
| [docs/09_rest_api.md](docs/09_rest_api.md) | REST API (Google API Design Guide): endpoints, errors, flexibility. |

---

## References

- [RecGPT paper](https://arxiv.org/abs/2506.06270)
- [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT)
- [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model)
- [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) (Steam splits)
