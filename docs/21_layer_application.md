# Layer 6: Application

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [20 Layer Recommendation](20_layer_recommendation.md).

---

## Problem or limitation

Eval and gRPC serving must be documented and testable; without a single surface (Eval, PredictionService, GRPCEndpoint), integration and QA are unclear.

---

## Proposed improvement

Document Layer 6 (Application): responsibility, public surface, and how to test. Eval and gRPC delegate to Serve.recommend.

Eval loads test cases and calls Serve.recommend to compute Hit@k, MRR, etc. PredictionService.Server handles gRPC Predict and delegates to Serve.recommend. GRPCEndpoint wires the server. **Public surface:** RecGPT.Eval.evaluate/3, RecGPT.Eval.load_test_cases/1, Recgpt.V1.PredictionService.Server (gRPC), RecGPT.GRPCEndpoint. **How to test:** eval_test.exs, prediction_service_test.exs. Stub Serve state for unit tests; integration tests use real stack.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [20 Layer Recommendation](20_layer_recommendation.md) - Serve.
- [06 Evaluation and testing](06_evaluation_and_testing.md).
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
