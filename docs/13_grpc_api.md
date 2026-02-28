# gRPC API (PredictionService)

The recommendation API is **gRPC-only**. The authoritative contract is [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). Optional HTTP transcoding (e.g., grpc-gateway) can use the `google.api.http` annotation on `Predict`.

---

## Service

| RPC        | Request          | Response           |
| ---------- | ----------------- | ------------------ |
| **Predict** | `PredictRequest`  | `PredictResponse`  |

- **Full name:** `recgpt.v1.PredictionService/Predict`
- **Optional REST:** `POST /v1/catalog:recommend` (body: JSON matching request fields) when transcoding is enabled.

---

## Messages

### PredictRequest

| Field               | Type             | Description |
| ------------------- | ---------------- | ----------- |
| `context_item_ids`  | `repeated int32` | **Required**, non-empty. Item IDs of the current context (e.g., recent clicks). |
| `max_results`       | `int32`         | Optional. Default 5, max 20. Number of recommended item IDs to return. |

### PredictResponse

| Field     | Type               | Description |
| --------- | ------------------- | ----------- |
| `item_ids` | `repeated int32`   | Ordered list of recommended item IDs. |
| `items`   | `repeated ItemSummary` | Same order: `item_id` and `display_name` for each. |

### ItemSummary

| Field          | Type     | Description |
| -------------- | -------- | ----------- |
| `item_id`      | `int32`  | Catalog item ID. |
| `display_name` | `string` | Human-readable label (from fixture/item text). |

### CatalogItem (future)

Reserved for future ingest/update (e.g., CatalogService). Not used by Predict.

| Field            | Type               | Description |
| ---------------- | ------------------ | ----------- |
| `item_id`        | `string`           | Primary key. |
| `slug`           | `string`           | URL-friendly identifier (e.g., market slug). |
| `content_jsonld` | `string`           | Dublin Core XMP JSON-LD. |
| `catalog_ids`    | `repeated string`  | Which catalog(s) this item belongs to. |

---

## Errors (gRPC status)

| Status            | When |
| ----------------- | ---- |
| `INVALID_ARGUMENT` | Invalid parameters (e.g., empty `context_item_ids`, invalid `max_results`). |
| `UNAVAILABLE`     | Service not ready (model/fixture not loaded). |

---

## Server

**Command:**

```bash
mix recgpt.serve --fixture <path> --ckpt <path> [--grpc-port 50051]
```

- **Fixture:** Path to `fixture.json` (num_items, token_id_list). Default: `data/serve_e2e_fixture.json` (or `RECGPT_FIXTURE`).
- **Checkpoint:** Path to checkpoint export dir (manifest + `.npy`). Default: `data/recgpt_ckpt_export` (or `RECGPT_CKPT_EXPORT`).
- **gRPC port:** Default 50051.

Fixture and checkpoint must exist; otherwise the server exits with an error.

**Implementation:** [RecGPT.GRPCEndpoint](../../lib/recgpt/grpc_endpoint.ex) runs `Recgpt.V1.PredictionService.Server`; the server delegates to `RecGPT.Serve.recommend/3`.

---

**Next:** [15_infrastructure_serving.md](15_infrastructure_serving.md).
