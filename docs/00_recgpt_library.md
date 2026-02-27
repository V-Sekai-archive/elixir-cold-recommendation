# RecGPT library — module reference

Reference for the **recgpt** package: modules, dependencies, and tests. For pipeline and evaluation concepts, see the linked docs below.

---

## Module overview

### Core: FSQ and embeddings

| Module                | Purpose                                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.FSQ**        | Finite Scalar Quantization: levels [8,8,8,6,5], 4 tokens per item, vocab 15360 + padding. `load_params/1`, `encode/2`, `codes_to_indices/1`, `indices_to_codes/2`.        |
| **RecGPT.FSQEncoder** | `encode_embeddings_to_token_id_list/3`: embeddings + FSQ params → list of 4-token lists. `load_embeddings_from_npy/1` for precomputed embeddings.                         |
| **RecGPT.Embedding**  | Text → 768-d via Bumblebee (sentence-transformers/all-mpnet-base-v2). `serving/0`, `encode_texts/1`, `encode_item_text_dict/1`, `save_embeddings/2`, `load_embeddings/1`. |

### Fixture and training data

| Module                  | Purpose                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.FixtureBuild** | Build fixture from items or embeddings. `build/3`, `build_from_embeddings/3`, `write_fixture/2`. Option `:fsq_dir` when FSQ params are not in the checkpoint. |
| **RecGPT.Training**     | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2`. Batch format matches Inference.                                                                   |

### Training loop

| Module               | Purpose                                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.AxonTrain** | `stream_batches/4`, `run/3`. Polaris optimizer; same forward and loss as Inference/Training. Options: `:iterations`, `:batch_size`, `:learning_rate`, `:log`. |

### Inference and serving

| Module               | Purpose                                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.Inference** | `forward/4` (logits at last position), `forward_full_sequence/4` (all positions, for training). Params from CheckpointLoader. |
| **RecGPT.Decode**    | `beam_search_top_k/4` → `{:ok, item_ids}` or `:not_found`.                                                                    |
| **RecGPT.Trie**      | Build trie from token_id_list for beam search.                                                                                |
| **RecGPT.Serve**     | `load_state/4` (opts: `:item_extra` for response enrichment), `recommend/3`, `search/3`, `item_ids_to_context_token_ids/3`.   |

### Evaluation

| Module          | Purpose                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- |
| **RecGPT.Eval** | `evaluate/3` (metrics: n, hit_at_k, mrr, random_hit_at_1, rejects_null), `load_test_cases/1`. |

### Checkpoint

| Module                      | Purpose                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------- |
| **RecGPT.CheckpointLoader** | `load_from_export/1` → `%{key => Nx.Tensor}`. Expects `manifest.json` and `.npy` files. |
| **RecGPT.CheckpointExport** | `write_export/2`. Writes manifest and one `.npy` per key.                               |
| **RecGPT.PtLoader**         | Load PyTorch `.pt` (zip format) for `mix recgpt.export_ckpt --from-pt`.                 |
| **RecGPT.ParamFlatten**     | Flatten nested state_dict when needed.                                                  |

### Data pipeline

| Module                             | Purpose                                                                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.Clickstream.Fetch**       | UCI Clickstream → SQLite; writes items and train/test/cold sequences. `run/2` (opts: `:max_train_sessions_for_cold`). `compute_cold_splits/4` (pure). |
| **RecGPT.Clickstream.CatalogItem** | Ecto schema for catalog items.                                                                                                                        |
| **RecGPT.Clickstream.EtnfEvent**   | Ecto schema for events (session_id, ord, item_id).                                                                                                    |
| **RecGPT.Xmp.DublinCore**          | Dublin Core IRIs and `context/0` for XMP JSON-LD.                                                                                                     |
| **RecGPT.Xmp.CatalogItemSchema**   | Grax schema for catalog item (DC properties).                                                                                                         |
| **RecGPT.Xmp.Jsonld**              | RDBMS → Grax → RDF → XMP JSON-LD. `from_catalog_item/1`, `validate_jsonld/1`, `to_xmp_jsonld_string/2`.                                               |
| **RecGPT.Repo**                    | Ecto repo (SQLite).                                                                                                                                   |

### HTTP and application

API: unified gRPC+REST ([13](13_grpc_rest_api.md), [14](14_api_schemas.md)). REST: [09](09_rest_api.md).

| Module                 | Purpose                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------- |
| **RecGPT.Serve.Plug**  | REST API router: GET /v1/catalog/items, POST /v1/catalog:recommend, GET /v1/health. |
| **RecGPT.Serve.REST**  | Request/response helpers and Google-style error body for the REST API.              |
| **RecGPT.Application** | Starts RecGPT.Repo.                                                                 |

---

## Dependencies

| Dependency                | Role                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Nx                        | Tensors.                                                                                                     |
| Axon                      | Model API (training loop is custom in AxonTrain).                                                            |
| Bumblebee (GitHub `main`) | MPNet text embeddings.                                                                                       |
| Jason, Npy                | JSON; checkpoint `.npy` load/save.                                                                           |
| RDF, JSON.LD, Grax        | XMP JSON-LD: Dublin Core struct mapping and validation (see [04](04_foss_datasets_etnf_dublin_core_xmp.md)). |
| Plug.Cowboy               | HTTP server.                                                                                                 |
| Ecto, ecto_sqlite3        | Clickstream database.                                                                                        |
| Req                       | HTTP (fetch_ckpt, Clickstream zip).                                                                          |
| Unpickler, Unzip          | PyTorch `.pt` loading.                                                                                       |
| PropCheck (dev/test)      | Property-based tests.                                                                                        |

See [mix.exs](../mix.exs).

---

## Tests

| Scope                              | Command                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Default                            | `mix test --no-start` (excludes embedding, integration, eval, e2e_serve, compare_python, pt_fixture). |
| Integration                        | `mix test --include integration`.                                                                     |
| Embedding                          | `mix test --include embedding` (downloads HF model; use long timeout).                                |
| Eval (fixture + ckpt + test files) | `mix test test/recgpt/eval_test.exs --include eval --include integration`.                            |
| PropCheck                          | `MIX_ENV=test mix run script/run_propcheck.exs` (see `test/recgpt/propcheck_test.exs.skip`).          |
| Parity constants                   | `mix test test/recgpt/parity_constants_test.exs`.                                                     |

Tests live in `test/recgpt/*_test.exs` and `test/support/recgpt/`.

---

## Training flow (summary)

1. **Data** — Fetch → items, train/test/cold sequences (see [07](07_steam_splits_and_pretraining.md), [08](08_pipeline_reference.md)).
2. **Fixture** — items → Embedding → FSQ → token_id_list → fixture.json (see [08](08_pipeline_reference.md)).
3. **Pretrain** — train_sequences + fixture + checkpoint → AxonTrain → updated checkpoint (see [08](08_pipeline_reference.md)).
4. **Eval** — fixture + checkpoint + test + cold_test → metrics (see [05](05_evaluation_and_testing.md), [08](08_pipeline_reference.md)).

**Zero-shot:** Pretrained checkpoint + fixture only (no training). **Trained:** Checkpoint fine-tuned on train split; same fixture and test sets. See [05 Evaluation and testing](05_evaluation_and_testing.md).

---

## See also

- [Documentation index](README.md) — All docs by topic.
- [04 FOSS datasets and Dublin Core XMP](04_foss_datasets_etnf_dublin_core_xmp.md) — Schema, XMP JSON-LD, RDF/Grax modules.
- [05 Evaluation and testing](05_evaluation_and_testing.md) — Zero-shot vs trained, null hypothesis, held-out eval.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Train/test/cold splits, pretrain-first.
- [08 Pipeline reference](08_pipeline_reference.md) — Commands and file layout.
- [09 REST API](09_rest_api.md) — Serve endpoints and request/response format.
- [RecGPT paper](https://arxiv.org/abs/2506.06270) · [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) · [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) · [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset)
