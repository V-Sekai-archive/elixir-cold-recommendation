# RecGPT

Elixir library for RecGPT-style recommendation: FSQ (Finite Scalar Quantization), text embeddings (MPNet via Bumblebee), and training data pipeline. No GenServer; use from any application (e.g. polymarket).

## Modules

- **RecGPT.FSQ** — FSQ quantizer (levels [8,8,8,6,5], 4 tokens per item, vocab 15360 + padding). Weights via `load_params/1` from exported VAE.
- **RecGPT.FSQEncoder** — `encode_embeddings_to_token_id_list/3`: (num_items, 768) → list of 4-token lists.
- **RecGPT.Embedding** — Text → 768-d (Bumblebee + sentence-transformers/all-mpnet-base-v2). `encode_texts/1`, `encode_item_text_dict/1`.
- **RecGPT.Training** — `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2` for training data and loss.
- **RecGPT.Serve** — HTTP server (port of Python `serve.py`): `load_state/3`, `recommend/3`, `search/3`. Run with `mix recgpt.serve [--port 8000]`.

## Clickstream PoC (UCI data → eval artifacts)

One command to fetch UCI Clickstream zip, run migrations, load SQLite, and write Steam-like artifacts: `items.json`, `train_sequences.json` (80% sessions for pretraining), and `test_sequences.json` (20% held-out). For best quality, pretrain on train then eval; see [docs/07_steam_splits_and_pretraining.md](docs/07_steam_splits_and_pretraining.md).

```bash
mix run -e "Application.ensure_all_started(:recgpt); RecGPT.Clickstream.Fetch.run()"
```

If you see "table already exists", remove `data/clickstream/recgpt.db` and run again. Next: build fixture (Embedding + FSQ from items), then `mix recgpt.eval`.

## HTTP server (serve.py port)

From `recgpt/` (or repo root; paths resolve to `data/` under cwd or parent):

```bash
mix recgpt.serve
# Optional: --port 8080 --fixture path/to/serve_e2e_fixture.json --ckpt path/to/recgpt_ckpt_export
```

Requires `data/serve_e2e_fixture.json` and `data/recgpt_ckpt_export/` (or set `RECGPT_FIXTURE` and `RECGPT_CKPT_EXPORT`). Serve E2E fixture and tests live in a separate repo; set RECGPT_FIXTURE to use that fixture. Endpoints:

- **POST /recommend** — body `{"item_ids": [1, 2, 3], "top_k": 5}` → `{"item_ids": [...], "item_texts": [...]}` (single best from beam search).
- **GET /search?q=...&limit=20** — catalog search by string.
- **GET /health** — `{"status": "ok"}`.

## Deps

Nx, Axon, Bumblebee (GitHub `main` for MPNet), Jason, Npy (for `load_embeddings_from_npy/1`), PropCheck (property-based tests, dev/test).

## Tests

From `recgpt/`:

```bash
mix test --exclude embedding
```

- Embedding tests load the HF model; run with `--include embedding` (and long timeout) if needed.
- **PropCheck** property tests: `mix test test/recgpt/propcheck_test.exs` (FSQ, Training, FSQEncoder).
- **Parity constants** (doc/code sync): `mix test test/recgpt/parity_constants_test.exs`.
- **Pipeline integration** (full flow): `mix test test/recgpt/pipeline_integration_test.exs` — embeddings → token_id_list → train batch → loss.
- **Serve E2E** (serve/predict flow): fixture and tests in a separate repo (see that repo’s serve_e2e project and `scripts/export_serve_e2e_fixture.py`). Set RECGPT_FIXTURE to use that fixture with mix recgpt.serve.

## Python comparison

Compare test is excluded by default (needs fixtures). Python scripts (e.g. `compare_recgpt_fsq.py`) may live in parent repo. With fixtures: `mix test test/recgpt/compare_test.exs --include compare_python`.

## Docs

- [Library documentation](docs/00_recgpt_library.md) — modules, deps, tests, training flow, links to repo RecGPT docs.
- [Python RecGPT parity progress](docs/01_python_recgpt_parity_progress.md) — task list, validation commands, PropCheck and parity constants tests.
- [Evaluation and testing](docs/05_evaluation_and_testing.md) — zero-shot vs trained, train/eval split (held-out eval), null-hypothesis rejection, test plan.
- [Steam dataset splits and pretraining](docs/07_steam_splits_and_pretraining.md) — regular + cold splits, pretraining-first for best quality.
