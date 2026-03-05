# Proposal: RecGPT library (module reference)

Sub-proposal of the [documentation index](README.md). One place to look up modules, dependencies, and tests.

---

## Problem or limitation

Contributors and users need a single reference for what each module does, which dependencies it uses, and how to run tests. Without it, discovery and onboarding depend on reading code or scattered docs.

---

## Proposed improvement

Maintain one **module reference** (this document) with overview tables by area, dependency list, test commands, and a short training flow. Pipeline and evaluation concepts link to their own proposals.

---

## Module overview

### Core: FSQ and embeddings

| Module                | Purpose                                                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **RecGPT.FSQ**        | Finite Scalar Quantization: levels [8,8,8,6,5], 4 tokens per item, vocab 15360 + padding. `load_params/1`, `encode/2`, `codes_to_indices/1`, `indices_to_codes/2`. |
| **RecGPT.FSQEncoder** | `encode_embeddings_to_token_id_list/3`: embeddings + FSQ params → list of 4-token lists.                                                                           |
| **RecGPT.Embedding**  | Text → 768-d via Bumblebee (sentence-transformers/all-mpnet-base-v2). `serving/0`, `encode_item_text_dict/1`.                                                      |

### Fixture and training data

| Module                  | Purpose                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **RecGPT.FixtureBuild** | Build fixture from items.json. `build/2`, `write_fixture/2`.                                |
| **RecGPT.Training**     | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2`. Batch format matches Inference. |

### Training loop

| Module                    | Purpose                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.AxonTrain**      | `stream_batches/4`, `run/3`. Polaris optimizer; same forward and loss as Inference/Training. Options: `:iterations`, `:batch_size`, `:learning_rate`, `:log`. |
| **RecGPT.PretrainRunner** | `run/1`. Library entry for pretrain pipeline (ckpt, fixture, train sequences, items → AxonTrain → export). Used by Mix task and StaffApi.                     |

### Inference and serving

| Module                     | Purpose                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.Inference**       | `forward/4` (logits at last position), `forward_full_sequence/4` (all positions, for training). Params from CheckpointLoader. |
| **RecGPT.FuxiLinearInference** | FuXi-Linear backbone (Retention + LinearTemporalChannel + LinearPositionalChannel). Same interface as Inference. Opts: `all_timestamps`, `chunk_size`. See [40](40_fuxi_linear_status.md). |
| **RecGPT.FuxiLinearInferenceDefn** | Defn JIT `forward_last_4_logits/4` for Serve when FuXi checkpoint. |
| **RecGPT.FuxiLinearInferenceParams** | Build defn params from FuXi checkpoint keys. |
| **RecGPT.Decode**          | `beam_search_top_k_spmd/8` (beam) and `lookahead_top_k/5` (MTP) → `{:ok, item_ids}` or `:not_found`. Strategy via `RECGPT_DECODE_STRATEGY`. |
| **RecGPT.Trie**      | Build trie from token_id_list for beam search.                                                                                |
| **RecGPT.Serve**     | `load_state/3`, `recommend/3`, `item_ids_to_context_token_ids/3`.                                                             |

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

| Module                 | Purpose                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **RecGPT.Steam.Fetch** | Steam test split from HuggingFace (hkuds/RecGPT_dataset); writes items.json, train/test/cold sequences. `run/1`. |

### Staff API

| Module                      | Purpose                                                                                                                                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.StaffApi**         | Behaviour + delegation. `list_items/1`, `get_item/1`, `upsert_items/1`, `sync_items_from_json/1`, `write_items_json/2`, `sync_sequences/1`, `build_fixture/3`, `write_fixture/2`, `pretrain/1`, `set_canonical_texts/1`. See [29 Staff API](29_staff_api.md). |
| **RecGPT.StaffApi.Default** | Default implementation (Sync, FixtureBuild, Repo, PretrainRunner).                                                                                                                                                                                            |

### gRPC API

API: gRPC only. Contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). See [01 gRPC API](01_grpc_api.md).

| Module                                 | Purpose                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------- |
| **RecGPT.GRPCEndpoint**                | gRPC endpoint; runs `Recgpt.V1.PredictionService.Server`.                 |
| **Recgpt.V1.PredictionService.Server** | gRPC server for Predict RPC; uses `RecGPT.PredictBatchCollector` → `RecGPT.Serve.recommend`. |

---

## Dependencies

| Dependency                | Role                                                    |
| ------------------------- | ------------------------------------------------------- |
| Nx                        | Tensors.                                                |
| Axon                      | Model API (training loop is custom in AxonTrain).       |
| Bumblebee (GitHub `main`) | MPNet text embeddings.                                  |
| Jason, Npy                | JSON; checkpoint `.npy` load/save.                      |
| (none for serve)          | `mix recgpt.serve` runs gRPC only; no HTTP REST server. |
| grpc, protobuf            | gRPC server and Protocol Buffers (PredictionService).   |
| Req                       | HTTP (fetch_steam).                                     |
| Unpickler, Unzip          | PyTorch `.pt` loading.                                  |

See [mix.exs](../mix.exs).

---

## Tests

| Scope                              | Command                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| Default                            | `mix test --no-start` (excludes integration, eval).                        |
| Integration                        | `mix test --include integration`.                                          |
| Eval (fixture + ckpt + test files) | `mix test test/recgpt/eval_test.exs --include eval --include integration`. |

Tests live in `test/recgpt/*_test.exs` and `test/support/recgpt/`.

---

## Training flow (summary)

1. **Data** — Fetch → items, train/test/cold sequences (see [07](07_steam_splits_and_pretraining.md), [02](02_pipeline_overview.md), [03](03_pipeline_steps.md)).
2. **Fixture** — items → Embedding → FSQ → token_id_list → fixture.json (see [02](02_pipeline_overview.md)).
3. **Pretrain** — train_sequences + fixture + checkpoint → AxonTrain → updated checkpoint (see [02](02_pipeline_overview.md)).
4. **Eval** — fixture + checkpoint + test + cold_test → metrics (see [06](06_evaluation_and_testing.md), [02](02_pipeline_overview.md), [03](03_pipeline_steps.md)).

**Zero-shot:** Pretrained checkpoint + fixture only (no training). **Trained:** Checkpoint fine-tuned on train split; same fixture and test sets. See [06 Evaluation and testing](06_evaluation_and_testing.md).

---

## Sub-proposals

- **Pipeline and data:** [07 Steam splits](07_steam_splits_and_pretraining.md), [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md).
- **Evaluation:** [06 Evaluation and testing](06_evaluation_and_testing.md).
- **Checkpoint:** [08 Checkpoint layout](08_recgpt_checkpoint_layout.md).
- **API:** [01 gRPC API](01_grpc_api.md).
- **Layer boundaries and test strategy:** [15 Layers and testing](15_layers_overview.md) — Maps module areas to layers (e.g. Core: FSQ and embeddings → Layer 2: Representation).

---

## See also

- [Documentation index](README.md) — Root proposal and all sub-proposals.
- [06 Evaluation and testing](06_evaluation_and_testing.md) — Zero-shot vs trained, null hypothesis, held-out eval.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Train/test/cold splits, pretrain-first.
- [02 Pipeline overview](02_pipeline_overview.md) — Commands and file layout.
- [01 gRPC API](01_grpc_api.md) — gRPC contract and serve.
- [RecGPT paper](https://arxiv.org/abs/2506.06270) · [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) · [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) · [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset)
