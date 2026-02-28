# Proposal: gRPC API (user-facing)

Sub-proposal of the [documentation index](README.md). **First** for users: the recommendation API is gRPC-only. This doc describes the contract, messages, errors, and how to run the server.

---

## Problem or limitation

Recommendation must be exposed via a stable, implementable contract. Without a single authoritative API spec, clients and servers diverge and integration is brittle.

---

## Proposed improvement

**gRPC-only** API. Authoritative contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). One RPC: **Predict** (PredictRequest â†’ PredictResponse). Run the server with `mix recgpt.serve`; fixture and checkpoint export dir are required.

---

## Service and RPC

- **Service:** `recgpt.v1.PredictionService`
- **RPC:** `Predict(PredictRequest) returns (PredictResponse)`
- **Request:** `context_item_ids` (repeated int32, required, non-empty), `max_results` (int32, optional, default 5, max 20).
- **Response:** `item_ids` (ordered recommended item IDs), `items` (repeated ItemSummary: `item_id`, `display_name`).

---

## Errors (gRPC status)

| Status             | When                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| `INVALID_ARGUMENT` | Invalid parameters (e.g. empty `context_item_ids`, invalid `max_results`). |
| `UNAVAILABLE`      | Service not ready (model/fixture not loaded).                              |

---

## Run the server

```bash
mix recgpt.serve --fixture <path> --ckpt <path> [--grpc-port 50051]
```

Defaults: `--fixture` â†’ `data/serve_e2e_fixture.json` (or `RECGPT_FIXTURE`), `--ckpt` â†’ `data/recgpt_ckpt_export` (or `RECGPT_CKPT_EXPORT`). Both fixture and checkpoint export directory are required.

### Quick test

With the server running (e.g. `mix recgpt.serve`):

```bash
grpcurl -plaintext -d '{"context_item_ids":[0,1], "max_results":5}' localhost:50051 recgpt.v1.PredictionService/Predict
```

---

## See also

- [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto) â€” Authoritative proto.
- [02 Pipeline overview](02_pipeline_overview.md) â€” How to produce fixture and checkpoint.
- [04 RecGPT library](04_recgpt_library.md) â€” Module reference (Serve, GRPCEndpoint).
