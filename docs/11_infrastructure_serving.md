# Proposal: Infrastructure and serving

Sub-proposal of the [documentation index](README.md). How to run and scale the recommendation server.

---

## Problem or limitation

Serving and deployment must be specified: how to run the server, what it depends on, and how it could scale (e.g. ETS, external inference). Without this, operators and SREs lack a single reference.

---

## Proposed improvement

**Current:** Single BEAM process; inference runs in-process with Nx. **Optional:** external inference (e.g., Triton) for GPU batching; multi-region/edge for lower latency. Document the run command, defaults, and optional scaling paths.

---

## In-process inference

At startup, `RecGPT.Serve.load_state/3` loads the checkpoint (`RecGPT.CheckpointLoader`, manifest + `.npy`), builds the trie from the fixture, and builds a logits function. `RecGPT.Inference.forward/4` runs the full forward pass (embedding, aux, GPT-2, head) in Nx. No separate inference server.

**Run the server:**

```bash
mix recgpt.serve --fixture <path> --ckpt <path> [--grpc-port 50051]
```

Defaults: `--fixture` → `data/serve_e2e_fixture.json` (or `RECGPT_FIXTURE`), `--ckpt` → `data/recgpt_ckpt_export` (or `RECGPT_CKPT_EXPORT`). Optional `--catalog` path to a catalog JSON (same shape as `items.json`) for item display names in gRPC responses. Both the fixture and checkpoint export directory are required. API contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto).

Nx can use the default backend (CPU) or Torchx if configured. For moderate traffic this is enough.

### Catalog storage (object-store semantics)

The catalog (item id and title for display names) is stored as JSON: shape `{"num_items", "items": [{"id", "title"}, ...]}` — see [04 Eval data shapes](04_eval_data_shapes.md#itemsjson). The path passed as `--catalog` (or the pipeline’s `items.json`) is the single source of truth. At serve time the catalog is read once and kept in memory; no runtime writes. When **writing** catalog from the app (e.g. a task that generates or updates items), use atomic replace: write to `path.tmp`, sync, then `File.rename(path.tmp, path)` so the visible file is never partial. This keeps storage SSD-stable and durable across restarts.

**Object-store options (Elixir/BEAM native):** Use **SQLite** for catalog storage so the repo stays stable. (1) **File-based** — built-in; JSON path, atomic replace for writes; no extra deps. (2) **CubDB** ([cubdb](https://hex.pm/packages/cubdb)) — pure Elixir; key-value, one key for catalog; add `{:cubdb, "~> 2.0"}`. (3) **Khepri** ([khepri](https://hex.pm/packages/khepri)) — RabbitMQ’s tree store (built on [Ra](https://hex.pm/packages/ra)); replicated, on-disk; put/get by path; add `{:khepri, "~> 0.17"}`. (4) **SQLite** — use SQLite (or whatever embedded store the project already uses) for catalog storage; describe in the docs how it fits with object-store semantics and the options above. All run on the BEAM.

#### Ra vs Khepri (RabbitMQ stack)

- **Ra** ([ra](https://hex.pm/packages/ra)) — Raft consensus library. You implement a **state machine** (via `ra_machine` behaviour); Ra replicates it across a cluster. Used by RabbitMQ for quorum queues, streams, and by Khepri as its engine. Use Ra when you need custom replicated state and are willing to implement the machine.
- **Khepri** ([khepri](https://hex.pm/packages/khepri)) — Tree-like replicated on-disk store **built on Ra**. It is the state machine that Ra replicates; you get a ready-made API: put/get/delete by path (e.g. `[:recgpt, :catalog]` → catalog blob). Use Khepri when you want a replicated key/path store without implementing Ra yourself.

For catalog storage, **Khepri** is the right fit (put one path, get at serve time). **Ra** is the lower-level layer; you’d only use it directly for a custom replicated store. 
---

## Sub-proposals

- **In-process inference** (above) — load_state, forward, run serve.
- **Optional:** Triton/edge (above) — External inference; multi-region.

---

**See also:** [docs README](README.md), [02 Pipeline reference](02_pipeline_reference.md).
