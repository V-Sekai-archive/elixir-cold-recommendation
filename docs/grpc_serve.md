# gRPC API (serve)

Service is gRPC-only. Contract: [priv/proto/recgpt/v1/recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). [01 gRPC API](01_grpc_api.md).

- **recgpt.v1.PredictionService/Predict** — Request: `context_item_ids`, `max_results`; response: `item_ids`, `items` (ItemSummary).

Errors use gRPC status (e.g. INVALID_ARGUMENT, UNAVAILABLE). See [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto).
