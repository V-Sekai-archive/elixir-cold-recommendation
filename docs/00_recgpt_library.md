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
| **RecGPT.FixtureBuild** | Build fixture from items.json. `build/3`, `write_fixture/2`. Option `:fsq_dir` when FSQ params are not in the checkpoint. |
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
| **RecGPT.Serve**     | `load_state/4`, `recommend/3`, `item_ids_to_context_token_ids/3`. |

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

### Data pipeline

| Module                 | Purpose                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| **RecGPT.Steam.Fetch** | Steam test split from HuggingFace (hkuds/RecGPT_dataset); writes items.json, train/test/cold sequences. `run/2`. |

### HTTP and application

API: gRPC only ([13](13_grpc_api.md), proto in `priv/proto/recgpt/v1/`). Contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto).

| Module                 | Purpose                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------- |
| **RecGPT.GRPCEndpoint** | gRPC endpoint; runs `Recgpt.V1.PredictionService.Server`.                          |
| **Recgpt.V1.PredictionService.Server** | gRPC server for Predict RPC; delegates to `RecGPT.Serve.recommend/3`.     |
| **RecGPT.Application** | Application callback (no supervised children).                                      |

---

## Dependencies

| Dependency                | Role                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Nx                        | Tensors.                                                                                                     |
| Axon                      | Model API (training loop is custom in AxonTrain).                                                            |
| Bumblebee (GitHub `main`) | MPNet text embeddings.                                                                                       |
| Jason, Npy                | JSON; checkpoint `.npy` load/save.                                                                           |
| (none for serve)          | `mix recgpt.serve` runs gRPC only; no HTTP REST server.                                                        |
| grpc, protobuf            | gRPC server and Protocol Buffers (PredictionService).                                                         |
| Req                       | HTTP (fetch_ckpt, fetch_steam).                                                                              |
| Unpickler, Unzip          | PyTorch `.pt` loading.                                                                                       |

See [mix.exs](../mix.exs).

---

## Tests

| Scope                              | Command                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Default                            | `mix test --no-start` (excludes integration, eval, e2e_serve). |
| Integration                        | `mix test --include integration`.                             |
| Eval (fixture + ckpt + test files) | `mix test test/recgpt/eval_test.exs --include eval --include integration`. |

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
- [05 Evaluation and testing](05_evaluation_and_testing.md) — Zero-shot vs trained, null hypothesis, held-out eval.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Train/test/cold splits, pretrain-first.
- [08 Pipeline reference](08_pipeline_reference.md) — Commands and file layout.
- [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto) — gRPC API contract.
- [RecGPT paper](https://arxiv.org/abs/2506.06270) · [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) · [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) · [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset)
