# Unified gRPC + REST API Design

One .proto defines the contract; `google.api.http` maps RPCs to REST; a gateway transcodes HTTP/JSON to gRPC so the same backend serves both. **Now:** REST only (predict, catalog list, health). gRPC and transcoding: intended extension.

---

## Design principles

- **Single source of truth:** The `.proto` file is the contract. REST paths and verbs come from `google.api.http` (see [AIP-127](https://google.aip.dev/127)); no separate REST spec.
- **Gateway transcoding:** REST clients send HTTP/JSON to the same host; the gateway (e.g. grpc-gateway or a Plug that inspects `Content-Type`) converts the request to the corresponding gRPC call and the response back to JSON. gRPC clients call the gRPC server directly (e.g. `application/grpc`).
- **Same backend:** Recommendation logic (trie, beam search, inference) lives in Elixir. Both gRPC and REST hit the same handlers.

---

## Current REST implementation

REST today: **Plug**/Cowboy, **`RecGPT.Serve.Plug`** and **`RecGPT.Serve.REST`**, backend **`RecGPT.Serve`**. Endpoints and errors: [09](09_rest_api.md). Full schema: [14](14_api_schemas.md). Run: `mix recgpt.serve --fixture path/to/fixture.json --ckpt path/to/ckpt_export` ([15](15_infrastructure_serving.md)).

---

## Adding gRPC and transcoding

1. **Define the full contract** in `.proto` with `google.api.http` on each RPC (see [14](14_api_schemas.md)).
2. **Add a gRPC server** (e.g. **grpc-elixir**, `GRPC.Server.Adapters.Cowboy.Handler`) in the same Elixir app. Implement the service callbacks that call into `RecGPT.Serve` (or equivalent).
3. **Unified gateway:** Use one Plug pipeline: if `Content-Type` is `application/grpc`, route to the gRPC handler; if `application/json` (or no gRPC headers), route to the REST/transcoding path. The transcoder turns the JSON body into the Protobuf request, calls the same gRPC handler, and converts the Protobuf response to JSON. That way REST and gRPC stay in sync by construction.
4. **Streaming:** Use **unary** RPCs for recommend and catalog list; use **bidirectional streaming** for event ingestion (and optionally catalog sync) so high-throughput clients avoid per-request connection overhead. In production, use client-side load balancing or a Layer 7 proxy and keep-alive so streams are not dropped.

**Next:** [14_api_schemas.md](14_api_schemas.md) — Full Protobuf schema and HTTP mappings.
