# Proposal: gRPC API (user-facing)

Sub-proposal of the [documentation index](README.md). **First** for users: the recommendation API is gRPC-only. This doc describes the contract, messages, errors, and how to run the server.

---

## Problem or limitation

Recommendation must be exposed via a stable, implementable contract. Without a single authoritative API spec, clients and servers diverge and integration is brittle.

---

## Proposed improvement

**gRPC-only** API. Authoritative contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). One RPC: **Predict** (PredictRequest → PredictResponse). Run the server with `mix recgpt.serve`; fixture and checkpoint export dir are required.

---

## Service and RPC

- **Service:** `recgpt.v1.PredictionService`
- **RPC:** `Predict(PredictRequest) returns (PredictResponse)`
- **Request:** `context_item_ids` (repeated int32, required, non-empty), `max_results` (int32, optional, default 5, max 20).
- **Response:** `item_ids` (ordered recommended item IDs), `items` (repeated ItemSummary: `item_id`, `display_name`).

The same server also runs **recgpt.v1.StaffService** (catalogues, sequences, fixture build, pretrain). Contract: [staff.proto](../priv/proto/recgpt/v1/staff.proto). See [29 Staff API](29_staff_api.md).

---

## Errors (gRPC status)

| Status             | When                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| `INVALID_ARGUMENT` | Invalid parameters (e.g. empty `context_item_ids`, invalid `max_results`). |
| `UNAVAILABLE`      | Service not ready (model/fixture not loaded).                              |

---

## Run the server

```bash
mix recgpt.serve --fixture <path> --ckpt <path> [--catalog <path>] [--grpc-port 50051]
```

Defaults: `--fixture` → `data/serve_e2e_fixture.json` (or `RECGPT_FIXTURE`), `--ckpt` → `data/recgpt_ckpt_export` (or `RECGPT_CKPT_EXPORT`). Both fixture and checkpoint export directory are required. Optional `--catalog` points to a JSON file with an `items` array (each with `id` and `title` or `text`); when provided, the Predict response uses those titles for `items[].display_name` so you get catalogue item names in recommendations.

### Quick test

**Option 1 — Mix task (uses proto file; no server reflection needed):**

Start the server in one terminal, then in another:

```bash
mix recgpt.grpc_curl
mix recgpt.grpc_curl --port 50051 --context 0,1 --max-results 10
```

**Option 2 — grpcurl directly (pass proto so server doesn’t need reflection):**

With the server running, use the sample request file so you don't need to escape JSON in the shell (avoids "Too many arguments" in PowerShell):

```bash
grpcurl -plaintext -import-path priv/proto -proto recgpt/v1/recommendation.proto -d @scripts/grpcurl_predict_request.json localhost:50051 recgpt.v1.PredictionService/Predict
```

The file `scripts/grpcurl_predict_request.json` contains `{"context_item_ids": [0], "max_results": 5}`. The `-d @path` form works in bash/WSL; in PowerShell `@` is interpreted as splat, so grpcurl gets the literal `@` and fails. Use the Mix task (Option 1) or pass the file contents in a variable:

**PowerShell (when Option 2 fails with "invalid character '@'"):**

```powershell
$body = Get-Content scripts/grpcurl_predict_request.json -Raw
grpcurl -plaintext -import-path priv/proto -proto recgpt/v1/recommendation.proto -d $body localhost:50051 recgpt.v1.PredictionService/Predict
```

Ensure the server is running (`mix recgpt.serve --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export --catalog data/steam/items.json`). Use port 50052 if you started the server with `--grpc-port 50052`.

---

## See also

- [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto) — Authoritative proto.
- [02 Pipeline overview](02_pipeline_overview.md) — How to produce fixture and checkpoint.
- [04 RecGPT library](04_recgpt_library.md) — Module reference (Serve, GRPCEndpoint).
