# API Schema: Unified gRPC + REST Contract

One .proto with **`google.api.http`** defines the contract; same RPCs as gRPC and REST. Current REST: [09](09_rest_api.md). Below: full schema — **EventService**, **PredictionService**, **CatalogService**.

---

## Service overview

| Service           | RPC               | REST (from google.api.http)                     | Implemented today  |
| ----------------- | ----------------- | ----------------------------------------------- | ------------------ |
| PredictionService | Predict           | POST /v1/...:predict (or /v1/catalog:recommend) | Yes (as recommend) |
| —                 | ListCatalogItems  | GET /v1/catalog/items                           | Yes                |
| CatalogService    | UpdateCatalogItem | POST /v1/catalog:update                         | No                 |
| —                 | RemoveCatalogItem | DELETE /v1/catalog/{item_id}                    | No                 |
| EventService      | StreamUserEvents  | — (streaming only)                              | No                 |
| —                 | WriteUserEvent    | POST /v1/events:write                           | No                 |
| Health            | Health            | GET /v1/health                                  | Yes                |

---

## Protocol Buffers (target)

Imports: `google/protobuf/timestamp.proto`, `google/api/annotations.proto`, `google/rpc/status.proto`, `google/protobuf/empty.proto` as needed.

### Prediction (recommend)

```protobuf
syntax = "proto3";
package recgpt.v1;

service PredictionService {
  rpc Predict (PredictRequest) returns (PredictResponse) {
    option (google.api.http) = {
      post: "/v1/catalog:recommend"
      body: "*"
    };
  }
}

message PredictRequest {
  repeated int32 context_item_ids = 1;
  int32 max_results = 2;  // default 5, max 20
}

message PredictResponse {
  repeated int32 item_ids = 1;
  repeated ItemSummary items = 2;
}

message ItemSummary {
  int32 item_id = 1;
  string display_name = 2;
}
```

### Catalog list and health

- **ListCatalogItems:** `ListRequest` (e.g. `q`, `page_size`) → `ListCatalogItemsResponse` (repeated ItemSummary). HTTP: `GET /v1/catalog/items`.
- **Health:** `Health(Empty) returns (HealthResponse)` with `GET /v1/health`.

### Event ingestion (target)

For real-time user events (views, clicks, cart, purchase) and optional catalog sync:

```protobuf
service EventService {
  rpc StreamUserEvents (stream UserEvent) returns (stream EventResponse);
  rpc WriteUserEvent (UserEvent) returns (EventResponse) {
    option (google.api.http) = {
      post: "/v1/events:write"
      body: "*"
    };
  }
}

enum EventType {
  EVENT_TYPE_UNSPECIFIED = 0;
  VIEW_ITEM = 1;
  ADD_TO_CART = 2;
  PURCHASE = 3;
}

message UserEvent {
  string event_id = 1;
  string user_id = 2;
  string session_id = 3;
  EventType event_type = 4;
  string item_id = 5;   // or int32 catalog item index, depending on schema
  float event_value = 6;
  google.protobuf.Timestamp timestamp = 7;
}

message EventResponse {
  string event_id = 1;
  google.rpc.Status status = 2;
}
```

### Catalog management (target)

For live catalog updates (IURS): new items are encoded (Embedding + FSQ) and the trie (or ETS-backed catalog) is updated without restart.

```protobuf
service CatalogService {
  rpc UpdateCatalogItem (CatalogItem) returns (google.protobuf.Empty) {
    option (google.api.http) = {
      post: "/v1/catalog:update"
      body: "*"
    };
  }
  rpc RemoveCatalogItem (RemoveItemRequest) returns (google.protobuf.Empty) {
    option (google.api.http) = {
      delete: "/v1/catalog/{item_id}"
    };
  }
}

message CatalogItem {
  string item_id = 1;
  string title = 2;
  string description = 3;
  repeated string categories = 4;
}

message RemoveItemRequest {
  string item_id = 1;
}
```

**Next:** [15_infrastructure_serving.md](15_infrastructure_serving.md).
