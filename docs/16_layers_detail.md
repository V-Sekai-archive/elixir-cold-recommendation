# Layers detail

Sub-proposal of the [documentation index](README.md). Per-layer modules, responsibility, and test strategy. Overview: [15 Layers overview](15_layers_overview.md).

---

## Layer 1: Artifacts

**What it does:** Read and write external artifacts: HuggingFace/Steam JSON, PyTorch `.pt` (zip), and the export directory (manifest.json + .npy). No RecGPT-specific logic beyond file layout.

**Public surface:** `Steam.Fetch.run/1`, `RecGPT.PtLoader`, `RecGPT.CheckpointLoader.load_from_export/1`, `RecGPT.CheckpointExport.write_export/2`.

**How to test:** Unit tests with temporary directories and fixture files; no dependency on other RecGPT layers.

---

## Layer 2: Representation

**What it does:** Turn item text into token IDs. Embedding (Bumblebee, MPNet) produces 768-d vectors; FSQ quantizes to 4 token IDs per item. No model weights beyond FSQ params.

**Public surface:** `RecGPT.Embedding.encode_item_text_dict/1`, `RecGPT.FSQ.load_params/1`, `RecGPT.FSQ.encode/2`, `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3`.

**How to test:** Unit tests with stub or real FSQ params; Embedding tests may require Bumblebee.

---

## Layer 3: Fixture

**What it does:** Build `fixture.json` (num_items, token_id_list) from items JSON and checkpoint (for FSQ params). Depends on Layer 1 (CheckpointLoader) and Layer 2 (Embedding, FSQEncoder).

**Public surface:** `RecGPT.FixtureBuild.build/2`, `RecGPT.FixtureBuild.write_fixture/2`.

**How to test:** Stub Embedding/CheckpointLoader or use real fixture files; see fixture_build_test.exs.

---

## Layer 4: Model

**What it does:** Forward pass (Inference), loss (Training), and training loop (AxonTrain). Params come from CheckpointLoader. Same forward/loss used for training and inference.

**Public surface:** `RecGPT.Inference.forward/4`, `RecGPT.Inference.forward_full_sequence/4`, `RecGPT.Training.build_train_batch/4`, `RecGPT.Training.loss_shifted_ce/2`, `RecGPT.AxonTrain.stream_batches/4`, `RecGPT.AxonTrain.run/3`.

**How to test:** inference_test.exs, training_test.exs, axon_train_test.exs. Stub checkpoint params for Inference.

---

## Layer 5: Recommendation

**What it does:** Trie from token_id_list; Decode runs beam search using a logits function (from Inference); Serve loads fixture + checkpoint, builds trie and get_logits, and exposes `recommend/3`.

**Public surface:** `RecGPT.Trie.build/1`, `RecGPT.Decode.beam_search_top_k/4`, `RecGPT.Serve.load_state/3`, `RecGPT.Serve.recommend/3`.

**How to test:** trie_test.exs, decode_test.exs, serve_test.exs. Stub get_logits for Trie/Decode; Serve tests can use stub state or full stack.

---

## Layer 6: Application

**What it does:** Eval loads test cases and calls Serve.recommend to compute Hit@k, MRR, etc. PredictionService.Server handles gRPC Predict and delegates to Serve.recommend. GRPCEndpoint wires the server.

**Public surface:** `RecGPT.Eval.evaluate/3`, `RecGPT.Eval.load_test_cases/1`, `Recgpt.V1.PredictionService.Server` (gRPC), `RecGPT.GRPCEndpoint`.

**How to test:** eval_test.exs, prediction_service_test.exs. Stub Serve state for unit tests; integration tests use real stack.

---

## See also

- [15 Layers overview](15_layers_overview.md) â€” Diagram and table.
- [04 RecGPT library](04_recgpt_library.md) â€” Module reference.
- [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md) â€”.
- [06 Evaluation and testing](06_evaluation_and_testing.md) â€”.
