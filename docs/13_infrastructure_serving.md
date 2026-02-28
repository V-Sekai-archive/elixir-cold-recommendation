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

**Health / readiness:** When running `mix recgpt.serve`, an HTTP health server listens on port **50052** by default (override with `--health-port` or `RECGPT_HEALTH_PORT`). Any request (e.g. `GET http://localhost:50052/`) returns **200 OK** when `serve_state` is loaded, or **503 Service Unavailable** before state is ready. Use this for Kubernetes readiness probes or load-balancer health checks. Set `--health-port 0` to disable the health server.

Nx can use the default backend (CPU) or Torchx if configured. For moderate traffic this is enough.

### Release and Docker

**Release:** Build with `mix release`. To run the gRPC server from a release, use `bin/recgpt eval "RecGPT.ReleaseTasks.serve()"`. Set env: `RECGPT_FIXTURE` (path to fixture JSON), `RECGPT_CKPT_EXPORT` (path to checkpoint export dir). Optional: `RECGPT_GRPC_PORT` (default 50051), `RECGPT_HEALTH_PORT` (default 50052; set to 0 to disable), `RECGPT_CATALOG` (path to catalog JSON). Paths must be absolute or relative to the release run directory.

**Docker:** A minimal Dockerfile can build the release and run the same eval command; set the env vars above when running the container. See the repo root `Dockerfile` (if present) for a multi-stage build that copies fixture and checkpoint into the image or mounts them at runtime.

### Catalog storage (object-store semantics)

The catalog (item id and title for display names) is stored as JSON: shape `{"num_items", "items": [{"id", "title"}, ...]}` — see [05 Eval data shapes](05_eval_data_shapes.md#itemsjson). The path passed as `--catalog` (or the pipeline's `items.json`) is the single source of truth. At serve time the catalog is read once and kept in memory; no runtime writes. When **writing** catalog from the app (e.g. a task that generates or updates items), use `RecGPT.Catalog.write!/2` (path, map or binary): it writes to `path.tmp`, syncs, then renames to `path` so the visible file is never partial and updates are SSD-safe. This keeps storage SSD-stable and durable across restarts.

**Object-store options (Elixir/BEAM native):** Use **SQLite** with Ecto as the primary catalog storage so the repo stays stable. (1) **File-based** — built-in; JSON path, atomic replace for writes; no extra deps. (2) **CubDB** ([cubdb](https://hex.pm/packages/cubdb)) — pure Elixir; key-value, one key for catalog; add `{:cubdb, "~> 2.0"}`. (3) **Khepri** ([khepri](https://hex.pm/packages/khepri)) — RabbitMQ's tree store (built on [Ra](https://hex.pm/packages/ra)); replicated, on-disk; put/get by path; add `{:khepri, "~> 0.17"}`. (4) **SQLite** — use SQLite (or whatever embedded store the project already uses) for catalog storage; describe in the docs how it fits with object-store semantics and the options above. All run on the BEAM.

#### Ra vs Khepri (RabbitMQ stack)

- **Ra** ([ra](https://hex.pm/packages/ra)) — Raft consensus library. You implement a **state machine** (via `ra_machine` behaviour); Ra replicates it across a cluster. Used by RabbitMQ for quorum queues, streams, and by Khepri as its engine. Use Ra when you need custom replicated state and are willing to implement the machine.
- **Khepri** ([khepri](https://hex.pm/packages/khepri)) — Tree-like replicated on-disk store **built on Ra**. It is the state machine that Ra replicates; you get a ready-made API: put/get/delete by path (e.g. `[:recgpt, :catalog]` → catalog blob). Use Khepri when you want a replicated key/path store without implementing Ra yourself.

For catalog storage, **Khepri** is the right fit (put one path, get at serve time). **Ra** is the lower-level layer; you'd only use it directly for a custom replicated store. 
---

## Sub-proposals

- **In-process inference** (above) — load_state, forward, run serve.
- **Optional:** Triton/edge (above) — External inference; multi-region.

---

**See also:** [docs README](README.md), [02 Pipeline overview](02_pipeline_overview.md), [03 Pipeline steps](03_pipeline_steps.md).
