# Layers overview

Sub-proposal of the [documentation index](README.md). Layer diagram and summary table. Per-layer docs: [16 Artifacts](16_layer_artifacts.md) | [17 Representation](17_layer_representation.md) | [18 Fixture](18_layer_fixture.md) | [19 Model](19_layer_model.md) | [20 Recommendation](20_layer_recommendation.md) | [21 Application](21_layer_application.md).

---

## Problem or limitation

Modules today form a dependency DAG but are flat under `RecGPT.*`. Tests already stub at boundaries (e.g. `build_stub_state` for Serve, stub state for Eval). There is no single document that defines layers or a testing strategy per layer.

---

## Proposed improvement

Define **six layers** (bottom to top): Artifacts, Representation, Fixture, Model, Recommendation, Application. Document for each: modules, responsibility, public surface, and how to test (what to stub, existing test files). **Dependency rule:** A layer only depends on layers below it. No circular deps. Each layer can be tested by stubbing the layer(s) below (or using real lower layers and only stubbing I/O).

**Isolate layers by freezing inputs (recommended):** Run the full pipeline once with full weights; capture the inputs at each layer boundary; then test or run each layer in isolation by feeding it only those frozen inputs. No GenServers, no stubbing—pure functions with fixed inputs. See [22 Freeze inputs for layer isolation](22_freeze_inputs_layer_isolation.md). Helper: `RecGPT.LayerFreeze.record_from_state/2`, `forward_model/2`, `recommend/3`.

**Each layer as a GenServer (optional):** Layers can be implemented as GenServer processes for process boundaries; trade-off is message-passing overhead (see [22](22_freeze_inputs_layer_isolation.md)). The current codebase keeps layers as stateless modules.

---

## Layer diagram

```mermaid
flowchart TB
  subgraph L1 [Layer 1: Artifacts]
    Steam[Steam.Fetch]
    PtLoader[PtLoader]
    CkptLoad[CheckpointLoader]
    CkptExport[CheckpointExport]
  end
  subgraph L2 [Layer 2: Representation]
    FSQ[FSQ]
    FSQEnc[FSQEncoder]
    Emb[Embedding]
  end
  subgraph L3 [Layer 3: Fixture]
    FixtureBuild[FixtureBuild]
  end
  subgraph L4 [Layer 4: Model]
    Inference[Inference]
    Training[Training]
    AxonTrain[AxonTrain]
  end
  subgraph L5 [Layer 5: Recommendation]
    Trie[Trie]
    Decode[Decode]
    Serve[Serve]
  end
  subgraph L6 [Layer 6: Application]
    Eval[Eval]
    Grpc[PredictionService]
  end
  FSQ --> FSQEnc
  CkptLoad --> FixtureBuild
  Emb --> FixtureBuild
  FSQ --> FixtureBuild
  FSQEnc --> FixtureBuild
  Inference --> Training
  Inference --> AxonTrain
  Training --> AxonTrain
  CkptLoad --> Inference
  CkptLoad --> Serve
  Inference --> Decode
  Trie --> Decode
  Decode --> Serve
  Serve --> Eval
  Serve --> Grpc
```

**Dependency rule:** A layer only depends on layers below it. No circular deps. Each layer can be tested by stubbing the layer(s) below (or using real lower layers and only stubbing I/O).

---

## Layers (bottom to top)

| Layer                 | Doc                              | Modules                                                           | Responsibility                                                                                                | Test strategy                                                                                  |
| --------------------- | -------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **1. Artifacts**      | [16](16_layer_artifacts.md)      | `Steam.Fetch`, `PtLoader`, `CheckpointLoader`, `CheckpointExport` | Read/write files and network: Steam JSON, `.pt`, export dir (manifest + .npy). No RecGPT business logic.      | Unit tests with temp files or fixtures; no other RecGPT modules.                               |
| **2. Representation** | [17](17_layer_representation.md) | `FSQ`, `FSQEncoder`, `Embedding`                                  | Text to vectors (Bumblebee) to token IDs (FSQ). No model, no checkpoint beyond FSQ params.                    | Unit tests with stub or real FSQ params; Embedding tests may need Bumblebee.                   |
| **3. Fixture**        | [18](18_layer_fixture.md)        | `FixtureBuild`                                                    | Items JSON + checkpoint (for FSQ params) to fixture.json (num_items, token_id_list).                          | Unit tests: stub Embedding/CheckpointLoader or use real files.                                 |
| **4. Model**          | [19](19_layer_model.md)          | `Inference`, `Training`, `AxonTrain`                              | Forward pass, loss, training loop. Params from checkpoint.                                                    | Unit tests: stub params for Inference; Training uses FSQ; AxonTrain uses Inference + Training. |
| **5. Recommendation** | [20](20_layer_recommendation.md) | `Trie`, `Decode`, `Serve`                                         | Trie from token_id_list; beam search (Decode) with get_logits from Inference; Serve = load_state + recommend. | Unit tests: Trie/Decode with stub get_logits; Serve with stub state or full stack.             |
| **6. Application**    | [21](21_layer_application.md)    | `Eval`, `Recgpt.V1.PredictionService.Server`, `GRPCEndpoint`      | Eval = metrics over test cases using Serve.recommend; gRPC = Predict RPC delegating to Serve.recommend.       | Unit tests: stub Serve state for Eval and PredictionService. Integration: real stack.          |

---

---

## See also

- [16 Artifacts](16_layer_artifacts.md) · [17 Representation](17_layer_representation.md) · [18 Fixture](18_layer_fixture.md) · [19 Model](19_layer_model.md) · [20 Recommendation](20_layer_recommendation.md) · [21 Application](21_layer_application.md).
- [22 Freeze inputs for layer isolation](22_freeze_inputs_layer_isolation.md) - Isolate layers with frozen inputs from a full-weights run.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
