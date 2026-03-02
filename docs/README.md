# Proposal: RecGPT Elixir library

This codebase is **one proposal**: an Elixir library for RecGPT-style sequential recommendation (FSQ, embeddings, training, inference, gRPC serving). Docs are in **user-facing order**: **gRPC API first**, then pipeline, library, data, eval, checkpoint, parity, and architecture. Each doc is a sub-proposal (problem → proposed improvement → sub-proposals). Start here, then follow links recursively.

**Standardized sections:** Every doc uses `## Problem or limitation`, `## Proposed improvement`, then any topic sections, and ends with `---` and `## See also`. Layer docs (16–21) use the same template; responsibility, public surface, and how to test are in the Proposed improvement section.

---

## Before you start

- **Project overview:** [../README.md](../README.md) — Quick start, pipeline summary, mix tasks, tests.
- **Split from root README:** [quick_start](quick_start.md) · [pipeline_summary](pipeline_summary.md) · [mix_tasks](mix_tasks.md) · [modules_overview](modules_overview.md) · [dependencies](dependencies.md) · [dev_container](dev_container.md) · [tests](tests.md) · [grpc_serve](grpc_serve.md) · [versioning_and_references](versioning_and_references.md).
- **Pipeline order:** 1 → 2 → 3 → 4 (Fetch → build_fixture → pretrain → eval). Fixture and checkpoint are required for pretrain and eval. To run the full flow (pretrain → catalogue → recommend), see [03 Pipeline steps — Run the whole thing](03_pipeline_steps.md#run-the-whole-thing-pretrain--catalogue--recommend).
- **Module reference:** [04 RecGPT library](04_recgpt_library.md) — Modules, dependencies, test tags.

### Pipeline overview

```mermaid
flowchart LR
  subgraph gen [1. Generate data]
    Fetch[Steam.Fetch.run]
    Fetch --> items[items.json]
    Fetch --> train[train_sequences.json]
    Fetch --> test[test_sequences.json]
    Fetch --> cold[cold_test_sequences.json]
    Fetch --> coldTrain[cold_train_sequences.json]
  end
  subgraph build [2. Build fixture]
    items --> Embed[Embedding + FSQ]
    Embed --> fixture[fixture.json]
  end
  subgraph pretrain [3. Pretrain]
    ckpt[Checkpoint export]
    train --> stream[AxonTrain.stream_batches]
    fixture --> stream
    stream --> run[AxonTrain.run]
    ckpt --> run
    run --> out[Updated export]
  end
  subgraph eval [4. Eval]
    out --> EvalTask[recgpt.eval]
    fixture --> EvalTask
    test --> EvalTask
    cold --> EvalTask
  end
```

---

## Problem or limitation

Sequential recommendation needs a **production-ready implementation** that: (1) matches the RecGPT paradigm (FSQ, hybrid attention, text-driven items); (2) runs entirely in Elixir/BEAM without Python at runtime; (3) provides a single reproducible pipeline from data to trained model and metrics; (4) exposes recommendations via a stable, implementable API (gRPC). Without a single specification and codebase that ties these together, implementations drift and evaluation is not comparable.

---

## Proposed improvement

Deliver one **RecGPT Elixir library** that:

- **API (first):** gRPC-only; `PredictionService.Predict` (PredictRequest → PredictResponse). Authoritative contract in `recommendation.proto`; serve via `mix recgpt.serve`.
- **Pipeline:** Fetch (Steam) → build fixture (Embedding + FSQ) → pretrain (AxonTrain) → eval (Hit@k, MRR, cold). All steps have commands and options; artifact layout is defined.
- **Checkpoint:** PyTorch `.pt` or in-memory params → export (manifest + .npy) → `CheckpointLoader` → `Inference`. Key mapping and loader contract are specified.
- **Evaluation:** Held-out test and cold-test; null hypothesis rejection (Hit@1 > random); zero-shot vs trained comparison.
- **Architecture:** In-process inference; trie + beam search; optional ETS path for scaling. No Python in-repo; parity validated by tests.

Design is **specific and actionable**: each sub-proposal below can be implemented or extended from the doc alone.

---

## Verification: problem solved

This codebase ties the four requirements together in one specification and implementation. You can verify each as follows.

| Requirement                                                           | How the codebase delivers                                                                                                                                                                                                                                | How to verify                                                                                                                                                                       |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(1) RecGPT paradigm** (FSQ, hybrid attention, text-driven items)    | `RecGPT.FSQ` / `FSQEncoder`, `RecGPT.Embedding` (Bumblebee/MPNet), `RecGPT.Inference` (bidirectional–causal), `RecGPT.Decode` (beam + trie). Pipeline: [02](02_pipeline_overview.md), [03](03_pipeline_steps.md); paradigm: [11](11_recgpt_paradigm.md). | Unit tests (FSQ, embedding, inference, decode); pipeline integration tests (`mix test`).                                                                                            |
| **(2) Elixir/BEAM only at runtime**                                   | No Python in-repo; `.pt` and pickle files are read via Elixir (Unpickler, zip). Bumblebee runs in the VM.                                                                                                                                                | `mix test`; no Python process; see [09](09_parity_overview.md), [10](10_parity_layers.md).                                                                                          |
| **(3) Single reproducible pipeline** (data → trained model → metrics) | Four steps with commands: Fetch → build_fixture → pretrain → eval. Artifact layout and options are defined.                                                                                                                                              | Run the pipeline: `mix recgpt.fetch_steam` → `mix recgpt.build_fixture` → `mix recgpt.pretrain` → `mix recgpt.eval`; see [02](02_pipeline_overview.md), [03](03_pipeline_steps.md). |
| **(4) Stable, implementable API** (gRPC)                              | `recommendation.proto` defines the contract; `PredictionService.Predict`; serve via `mix recgpt.serve`.                                                                                                                                                  | Unit tests for Predict (validation, errors); full-flow test (load_state → predict); manual: `grpcurl` per [01](01_grpc_api.md#quick-test).                                          |

**End-to-end:** A single test exercises the full stack in-process: `Serve.load_state` (fixture + checkpoint) → state in application env → `PredictionService.Server.predict` → valid `PredictResponse`. That confirms data → model → API is wired correctly in this codebase.

### How to validate (backwards from 23)

To validate that the **library works when you use it**, run the QA checklist ([23](23_quality_assurance.md)). Steps 1–5 (compile, format, Credo, unit tests, Dialyzer) need no pipeline; they confirm the codebase builds and passes tests. Step 6 (Steam top-k) requires running the pipeline (fetch*steam, build_fixture, pretrain) and setting RECGPT*\* env; when it passes, the library behaves correctly with real data. That checklist is the single pass/fail gate for use.

Optionally you can also run the full pipeline yourself, run `mix recgpt.serve` and call Predict (e.g. via grpcurl per [01](01_grpc_api.md#quick-test)), or run eval with your own fixture and checkpoints — all of these exercise the library in use.

### Feature status

| Source                                                        | Done | Total | Notes                                                              |
| ------------------------------------------------------------- | ---- | ----- | ------------------------------------------------------------------ |
| [22 Top-tier recommendations](22_top_tier_recommendations.md) | 6    | 6     | All recommended improvements done.                                 |
| [10 Parity by layer](10_parity_layers.md)                     | 31   | 32    | 1 optional (numerical parity: Elixir forward vs reference logits). |

---

## Sub-proposals (user-facing order)

| #   | Proposal                                                                   | Problem / limitation                                                      | Sub-proposals                                                                                               |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 01  | [01_grpc_api.md](01_grpc_api.md)                                           | Recommendation must have a stable, implementable contract.                | Predict RPC; Errors; Run the server.                                                                        |
| 02  | [02_pipeline_overview.md](02_pipeline_overview.md)                         | Pipeline order and Step 1 (generate data).                                | Overview; Step 1. See [03](03_pipeline_steps.md) for steps 2–4, serve, layout.                              |
| 03  | [03_pipeline_steps.md](03_pipeline_steps.md)                               | Steps 2–4, serve, checkpoint setup, file layout.                          | Build fixture; Pretrain; Eval; Optional serve; Env vars.                                                    |
| 04  | [04_recgpt_library.md](04_recgpt_library.md)                               | Need a single module/dependency reference for the package.                | By area: FSQ, Fixture, Training, Inference, Serve, Eval, Checkpoint, Data.                                  |
| 05  | [05_eval_data_shapes.md](05_eval_data_shapes.md)                           | Tests and tools need canonical JSON shapes.                               | Per-file: test_sequences, cold_test, items, fixture, train_sequences, cold_train.                           |
| 06  | [06_evaluation_and_testing.md](06_evaluation_and_testing.md)               | Need to measure accuracy and reject the null baseline.                    | Zero-shot vs trained; Null hypothesis; Held-out eval; Commands.                                             |
| 07  | [07_steam_splits_and_pretraining.md](07_steam_splits_and_pretraining.md)   | Train/test/cold semantics and artifact layout must be clear.              | Artifact table; cold split definition.                                                                      |
| 08  | [08_recgpt_checkpoint_layout.md](08_recgpt_checkpoint_layout.md)           | RecGPT weights are PyTorch; Elixir needs export layout and loader.        | Components; Export; Mapping to inference.                                                                   |
| 09  | [09_parity_overview.md](09_parity_overview.md)                             | Parity at a glance and reference mapping.                                 | At a glance; mapping; summary. See [10](10_parity_layers.md) for per-layer.                                 |
| 10  | [10_parity_layers.md](10_parity_layers.md)                                 | Per-layer parity task lists and validation.                               | Embeddings; FSQ; Training; Forward; Decode; Checkpoint; E2E.                                                |
| 11  | [11_recgpt_paradigm.md](11_recgpt_paradigm.md)                             | Algorithmic foundations must be documented.                               | FSQ and semantic tokenization; Hybrid attention; Pipeline and modules.                                      |
| 12  | [12_dynamic_state_ets.md](12_dynamic_state_ets.md)                         | Decoding must be catalog-aware; scaling may need ETS.                     | Trie; Beam search; Future ETS.                                                                              |
| 13  | [13_infrastructure_serving.md](13_infrastructure_serving.md)               | Serving and deployment must be specified.                                 | In-process inference; Run serve; Optional Triton/edge.                                                      |
| 14  | [14_architecture_references.md](14_architecture_references.md)             | Claims and design must be citable.                                        | Works cited (RecGPT, beam/trie, ETS, gRPC).                                                                 |
| 15  | [15_layers_overview.md](15_layers_overview.md)                             | Layer diagram and summary table.                                          | Six layers; dependency rule. See [16](16_layer_artifacts.md)-[21](21_layer_application.md) for per-layer.   |
| 16  | [16_layer_artifacts.md](16_layer_artifacts.md)                             | Layer 1: Artifacts.                                                       | Steam.Fetch, PtLoader, CheckpointLoader, CheckpointExport.                                                  |
| 17  | [17_layer_representation.md](17_layer_representation.md)                   | Layer 2: Representation.                                                  | FSQ, FSQEncoder, Embedding.                                                                                 |
| 18  | [18_layer_fixture.md](18_layer_fixture.md)                                 | Layer 3: Fixture.                                                         | FixtureBuild.                                                                                               |
| 19  | [19_layer_model.md](19_layer_model.md)                                     | Layer 4: Model.                                                           | Inference, Training, AxonTrain.                                                                             |
| 20  | [20_layer_recommendation.md](20_layer_recommendation.md)                   | Layer 5: Recommendation.                                                  | Trie, Decode, Serve.                                                                                        |
| 21  | [21_layer_application.md](21_layer_application.md)                         | Layer 6: Application.                                                     | Eval, PredictionService, GRPCEndpoint.                                                                      |
| 22  | [22_top_tier_recommendations.md](22_top_tier_recommendations.md)           | Elevate the library to production-grade quality.                          | Typespecs/Dialyzer; integration test; health; property tests; benchmarks; release.                          |
| —   | [embedding_vs_eval.md](embedding_vs_eval.md)                               | Divide: embeddings vs eval.                                               | Generating embeddings (parity, .npy) vs testing recommendation performance (Hit@k, MRR, serve).             |
| —   | [22_freeze_inputs_layer_isolation.md](22_freeze_inputs_layer_isolation.md) | Isolate layers with frozen inputs (unit/property tests).                  | Problem/Proposed; Unit tests and property testing; Layer boundaries; Implementation; See also.              |
| 23  | [23_quality_assurance.md](23_quality_assurance.md)                         | Run the QA checklist before merge or release.                             | Compile, format, Credo, unit tests, Dialyzer; Steam top-k; CI.                                              |
| 24  | [24_first_step_plan.md](24_first_step_plan.md)                             | First step: Steam test vectors (baseline).                                | Why Steam first; get data, build fixture, run eval; outcome.                                                |
| 25  | [25_mvp_guard_rails.md](25_mvp_guard_rails.md)                             | Keep rope bridge on track.                                                | Guard rails (tombstones); no multi-rank/sharding until minimal loop closed.                                 |
| 26  | [26_embedding_mismatch.md](26_embedding_mismatch.md)                       | Embedding parity gap and workaround.                                      | Text format; compare_embeddings; use dataset .npy.                                                          |
| 28  | [28_thirdparty_vs_elixir_parity.md](28_thirdparty_vs_elixir_parity.md)     | Parity with released model (dataset .npy + VAE).                          | Embeddings and FSQ sources; use dataset .npy + VAE for FSQ; Elixir-only path.                               |
| 29  | [29_staff_api.md](29_staff_api.md)                                         | Staff API for catalogues, sequences, fixture, pretrain.                   | RecGPT.StaffApi behaviour; list/upsert items; sync sequences; build_fixture; pretrain; set_canonical_texts. |
| 30  | [30_waffle_ecto_usage.md](30_waffle_ecto_usage.md)                         | Blob storage with Ecto and optional object store.                         | waffle_ecto + Waffle: schema, cast_attachments, local/S3 config.                                            |
| 31  | [31_ycsb_storage_classification.md](31_ycsb_storage_classification.md)     | Classify storage by YCSB workload types and throughput.                   | YCSB A–F; database/store fit; RecGPT artifact mapping.                                                      |
| 32  | [32_spmd_decode_flow.md](32_spmd_decode_flow.md)                           | Minimize CPU–device sync in beam search; keep trie and scoring on device. | Trie tensors; SPMD beam search; single sync; lib/ modules (Trie, Decode, Serve).                            |

---

## Quick reference (actionable)

| I want to…                                         | See                                                                                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Call the recommendation API (gRPC)**             | [01 gRPC API](01_grpc_api.md), [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto), `mix recgpt.serve`                           |
| Run the full pipeline                              | [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md), [../README.md](../README.md#pipeline)                 |
| Find a module's purpose and API                    | [04 RecGPT library](04_recgpt_library.md)                                                                                                         |
| Generate or use test/fixture JSON                  | [05 Eval data shapes](05_eval_data_shapes.md)                                                                                                     |
| Run eval and interpret metrics                     | [06 Evaluation and testing](06_evaluation_and_testing.md)                                                                                         |
| Separate embeddings from eval (divide)             | [embedding_vs_eval.md](embedding_vs_eval.md) — generating embeddings vs testing recommendation performance                                        |
| Understand cold vs regular splits                  | [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md)                                                                             |
| Export or load a checkpoint                        | [08 Checkpoint layout](08_recgpt_checkpoint_layout.md)                                                                                            |
| Use SQLite/Ecto for catalog storage                | [13 Infrastructure](13_infrastructure_serving.md#catalog-storage-object-store-semantics)                                                          |
| Store blobs with Ecto (local or S3/GCS)            | [30 waffle_ecto usage](30_waffle_ecto_usage.md) — waffle_ecto + Waffle for attachments and optional object store.                                 |
| Classify storage by YCSB types and throughput      | [31 YCSB storage classification](31_ycsb_storage_classification.md) — workload types A–F, database fit, RecGPT artifact mapping.                  |
| Design catalog/DB schema (ETNF)                    | [ETNF database design](etnf_database_design.md)                                                                                                   |
| Understand layers and test strategy                | [15 Layers overview](15_layers_overview.md), [16](16_layer_artifacts.md)–[21](21_layer_application.md) layer docs.                                |
| Isolate layers with frozen inputs                  | [22 Freeze inputs for layer isolation](22_freeze_inputs_layer_isolation.md)                                                                       |
| Make the library top tier                          | [22 Top-tier recommendations](22_top_tier_recommendations.md)                                                                                     |
| Run the QA checklist                               | [23 Quality assurance](23_quality_assurance.md)                                                                                                   |
| First step (Steam baseline), MVP guard rails       | [24 First step plan](24_first_step_plan.md), [25 MVP guard rails](25_mvp_guard_rails.md); one-shot: `mix recgpt.first_step` (requires checkpoint) |
| Embedding parity and workaround                    | [26 Embedding mismatch](26_embedding_mismatch.md)                                                                                                 |
| Parity with released model (dataset .npy + VAE)    | [28 Thirdparty vs Elixir parity](28_thirdparty_vs_elixir_parity.md) — use `--embeddings-npy` and `--vae-ckpt` when building fixture.              |
| **Build a staff API (catalogues, pretrain, etc.)** | [29 Staff API](29_staff_api.md) — RecGPT.StaffApi: list/upsert items, sync sequences, build_fixture, pretrain.                                    |
| Understand SPMD decode (trie tensors, single sync) | [32 SPMD decode flow](32_spmd_decode_flow.md) — Trie.to_tensors, Decode.beam_search_top_k_spmd, Serve.recommend.                                  |
| Read the architecture blueprint                    | [11 Paradigm](11_recgpt_paradigm.md), [12 Dynamic state](12_dynamic_state_ets.md), [13 Infrastructure](13_infrastructure_serving.md)              |

---

## See also

- [01 gRPC API](01_grpc_api.md) — Recommendation API contract and server.
- [02 Pipeline overview](02_pipeline_overview.md) — Pipeline order and Step 1.
- [04 RecGPT library](04_recgpt_library.md) — Module reference.
- [15 Layers overview](15_layers_overview.md) — Layer diagram and table.
- [22 Freeze inputs for layer isolation](22_freeze_inputs_layer_isolation.md) — Unit/property testing with frozen inputs.
- [24 First step plan](24_first_step_plan.md), [25 MVP guard rails](25_mvp_guard_rails.md), [26 Embedding mismatch](26_embedding_mismatch.md), [28 Thirdparty vs Elixir parity](28_thirdparty_vs_elixir_parity.md).
- [ETNF database design](etnf_database_design.md) — Essential Tuple Normal Form for catalog/embedding schemas.
