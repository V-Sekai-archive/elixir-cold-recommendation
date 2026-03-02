# Staff API service

Sub-proposal of the [documentation index](README.md). **Staff/admin API** for creating and editing catalogues, sequences, building fixture, and running pretraining. Exposed via **gRPC** (`recgpt.v1.StaffService`) on the same endpoint as `PredictionService`.

---

## Problem or limitation

Staff need a single service layer to manage catalogues (items), sequences (train/test), fixture build, and pretraining. Without a defined contract and implementation, staff APIs would duplicate logic from Mix tasks and scatter calls across Sync, FixtureBuild, and Repo.

---

## Proposed improvement

**gRPC:** `recgpt.v1.StaffService` runs on the same server as `recgpt.v1.PredictionService` (`mix recgpt.serve`). Contract: [staff.proto](../priv/proto/recgpt/v1/staff.proto). RPCs: ListItems, GetItem, UpsertItems, SyncItemsFromJson, WriteItemsJson, SyncSequences, BuildFixture, WriteFixture, Pretrain, SetCanonicalTexts.

**Library:** **RecGPT.StaffApi** is a behaviour plus default implementation. The gRPC server delegates to `RecGPT.StaffApi.*`. You can also call `RecGPT.StaffApi.*` from code or a custom HTTP wrapper.

Configure a custom implementation with:

```elixir
config :recgpt, :staff_api_impl, MyApp.StaffApi.Custom
```

Default: `RecGPT.StaffApi.Default`.

---

## Operations

| Operation | Function | Description |
| --------- | -------- | ----------- |
| **Catalog (items)** | `list_items(:db)` | List items from DB (Item table). |
| | `list_items({:path, path})` | List items from a JSON file. |
| | `get_item(item_id)` | Get one item by id. |
| | `upsert_items(entries)` | Insert or update items; `entries` = `[%{item_id: id, title: title}, ...]`. |
| | `sync_items_from_json(path)` | Replace catalog from `items.json` (clears catalog tables, inserts items; run build_fixture to refresh tokens). |
| | `write_items_json(path, items)` | Write items to JSON (`%{"items" => ..., "num_items" => n}`). |
| **Sequences** | `sync_sequences(data_dir)` | Sync train, cold_train, test, cold_test from `data_dir` (e.g. `data/steam`). |
| **Fixture** | `build_fixture(items_path, ckpt_dir, opts)` | Build fixture (num_items, token_id_list). Options: `:canonical_texts`, `:limit`, `:vae_ckpt`, `:embeddings_npy`, `:sqlite`. |
| | `write_fixture(fixture, path)` | Write fixture map to JSON path. |
| **Pretrain** | `pretrain(opts)` | Run pretraining. Options: `:ckpt_dir` (or `:ckpt`), `:fixture_path` (or `:fixture`), `:train_path`, `:items_path`, `:out_dir` (or `:out`), `:iterations`, `:batch_size`, `:learning_rate`, `:log`, `:log_interval_sec`, `:limit`, `:resource_check_opts`. |
| **Canonical texts** | `set_canonical_texts(entries)` | Replace canonical_item_texts; `entries` = `[%{item_id: id, text: binary}, ...]` (for RecGPT parity). |

All functions return `{:ok, result}` or `:ok` on success, and `{:error, reason}` on failure.

---

## SPMD compatibility

The API is designed so it can be used in an **SPMD** (Single Program Multiple Data) deployment: the same program runs on every rank, with rank-scoped data (paths, catalog, fixture). The current implementation is single-rank; the contract is multi-rank ready.

- **Explicit scope:** All operations take explicit parameters (path, data_dir, items_path, ckpt_dir, out_dir). There is no implicit global catalog or fixture in the contract; each rank can pass rank-specific paths (e.g. `data_dir = "data/steam/rank_#{rank}"`).
- **Optional rank in gRPC:** Every Staff request and `PredictRequest` include an optional `rank` (int32, 0-based). Single-rank servers ignore it. Multi-rank deployments can use it to route the request to the correct shard or to derive rank-scoped paths (e.g. prefix paths with `rank_N/`).
- **Determinism:** For a given (rank, paths, inputs), the result is deterministic. Metrics (e.g. eval) can be reduced across ranks (sum hits, sum MRR numerator/denominator) when moving to multi-rank eval.
- **No process-global state in the contract:** The library may use process-local state (e.g. Repo, serve_state); the *API contract* does not assume a single global instance. Each rank can run the same binary with rank-specific config or paths.

See [25 MVP guard rails](25_mvp_guard_rails.md): we do not implement multi-rank execution yet; the API is compatible so that when guard rails are lifted, the same proto and behaviour can be used.

---

## gRPC quick test

With the server running (`mix recgpt.serve`):

```bash
# List items from DB (path empty => from_db)
grpcurl -plaintext -d '{"path":""}' localhost:50051 recgpt.v1.StaffService/ListItems

# Get one item
grpcurl -plaintext -d '{"item_id":0}' localhost:50051 recgpt.v1.StaffService/GetItem

# Sync sequences from a data directory
grpcurl -plaintext -d '{"data_dir":"data/steam"}' localhost:50051 recgpt.v1.StaffService/SyncSequences
```

## Example usage (library)

```elixir
# List items from DB
{:ok, items} = RecGPT.StaffApi.list_items(:db)

# Upsert a few items
:ok = RecGPT.StaffApi.upsert_items([
  %{item_id: 0, title: "Game A"},
  %{item_id: 1, title: "Game B"}
])

# Sync sequences from a data directory
:ok = RecGPT.StaffApi.sync_sequences("data/steam")

# Build fixture and write to path
{:ok, fixture} = RecGPT.StaffApi.build_fixture("data/steam/items.json", "data/recgpt_ckpt_export", canonical_texts: true)
:ok = RecGPT.StaffApi.write_fixture(fixture, "data/steam/fixture.json")

# Run pretraining
:ok = RecGPT.StaffApi.pretrain(
  ckpt_dir: "data/recgpt_ckpt_export",
  fixture_path: "data/steam/fixture.json",
  train_path: "data/steam/train_sequences.json",
  items_path: "data/steam/items.json",
  out_dir: "data/ckpt_after_pretrain",
  iterations: 100,
  batch_size: 8
)
```

---

## PretrainRunner

Pretraining logic lives in **RecGPT.PretrainRunner**. The Mix task `mix recgpt.pretrain` calls `PretrainRunner.run/1`. StaffApi.pretrain/1 converts option names (e.g. `:ckpt` → `:ckpt_dir`) and calls `PretrainRunner.run/1`. Use `PretrainRunner.run/1` directly from code when you already have normalized options.

---

## See also

- [01 gRPC API](01_grpc_api.md) — Recommendation API (PredictionService); same endpoint as StaffService.
- [staff.proto](../priv/proto/recgpt/v1/staff.proto) — Staff gRPC contract.
- [03 Pipeline steps](03_pipeline_steps.md) — Build fixture, pretrain, eval, serve.
- [04 RecGPT library](04_recgpt_library.md) — Module reference (Sync, FixtureBuild, Serve).
- [24 First step plan](24_first_step_plan.md) — Steam baseline and catalogue recommendation.
