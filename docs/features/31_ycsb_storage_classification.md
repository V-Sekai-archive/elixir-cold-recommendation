# YCSB storage classification — workload types, throughput, and database fit

Sub-proposal of the [documentation index](README.md). **Technique:** classify storage and access patterns using [YCSB](https://github.com/brianfrankcooper/YCSB) (Yahoo! Cloud Serving Benchmark) workload types, then map existing databases and stores to those workloads and to rough throughput so you can choose storage by access pattern.

---

## Problem or limitation

Choosing a database or store (ETS, SQLite, Redis, object store, file) without a clear **access-pattern** lens leads to mismatches: e.g. using a write-optimized store for read-heavy lookups, or expecting high request-rate semantics from an object store. We need a **repeatable technique** to (1) classify workloads by read/write/scan mix, (2) compare stores by workload fit and throughput, and (3) apply the result to concrete artifacts (catalog, fixture, checkpoint, train data).

---

## Proposed improvement

Use **YCSB workload types (A–F)** as the classification scheme:

1. **Classify** each storage need by its dominant operation mix (point read, update, scan, read-modify-write, etc.).
2. **Map** existing databases and stores to which workloads they suit and to order-of-magnitude throughput.
3. **Apply** the mapping when designing or documenting systems (e.g. RecGPT pipeline, catalog, artifacts).

Throughput numbers here are **estimates** for comparison; run your own benchmarks for production sizing.

---

## YCSB workload types (quick reference)

| Type  | Name / mix        | Description                                                                            |
| ----- | ----------------- | -------------------------------------------------------------------------------------- |
| **A** | Update heavy      | 50% read, 50% update. Point reads and point updates; mixed read/write.                 |
| **B** | Read mostly       | 95% read, 5% update. Point reads dominate; occasional updates (e.g. cache refresh).    |
| **C** | Read only         | 100% read. Point reads only; no writes on the hot path.                                |
| **D** | Read latest       | 95% read, 5% insert. Reads tend to hit recently inserted keys (e.g. time-ordered).     |
| **E** | Scan              | 95% scan (range), 5% insert. Range or scan operations over many keys.                  |
| **F** | Read-modify-write | 50% read, 50% read-modify-write. Read key, compute, write back (e.g. counters, state). |

**How to classify:** Ask: What is the **dominant** operation on this store? (Point read by key → B or C. Bulk read one blob → C. Stream/scan over keys → E. Update in place → A or F.) If the hot path has **no writes**, prefer C. If it’s **refresh or occasional update**, B. If it’s **read then write back**, F.

---

## Technique: classify then map

1. **Identify the access pattern** for the data (e.g. “lookup by item_id,” “stream batches,” “bulk write at save”).
2. **Assign a YCSB type** (usually one primary: C for read-only, B for read-mostly, F for read-modify-write, A for update-heavy, E for scan, D for read-latest).
3. **Look up** which stores are strong for that type (table below).
4. **Check throughput** and operational constraints (single node vs distributed, persistence, network).

---

## Existing databases and stores: workload fit and throughput

Throughput is order-of-magnitude; actual numbers depend on payload size, hardware, and configuration. Use for **comparison**, not as a guarantee.

| Database / store          | Strong workloads                    | Why                                                                                                                                                                                       | Weaker fit                                                                                                                                             | Throughput (est.)                  |
| ------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| **Redix (Redis)**         | **B**, **C**, **A**                 | In-memory; sub-ms point GET/SET; pipelining for batch reads. Good for mixed read/update when data fits in RAM.                                                                            | **E** (scan): range/scan not primary. **D** (read latest): no native “recent key” unless you maintain it.                                              | ~50k ops/sec per core              |
| **ETS**                   | **B**, **C**, **F**                 | In-process; no serialization or network. Fastest point lookups and in-memory updates; ideal for C and F when the store is an ETS table.                                                   | **A** (update heavy to disk): ETS is in-memory; persistence is separate. **D**/**E**: possible with `:ordered_set` or iteration but not main strength. | ~500k per process                  |
| **SQLite (Ecto)**         | **B**, **C**                        | Single file; low latency for local reads; excellent for read-heavy and point lookups; ACID with one writer.                                                                               | **A**: single-writer can bottleneck under high write concurrency. **E**: OK but not optimized for huge range scans.                                    | ~10k TPS mixed                     |
| **PostgreSQL**            | **B**, **C**, **A**, **F**          | ACID; multi-connection; good for read-heavy and mixed; connection pooling (e.g. PgBouncer).                                                                                               | **E**: full table scans costly without indexing; very high write contention can bottleneck.                                                            | ~10k TPS mixed                     |
| **FoundationDB**          | **A**, **D**, **E**, **F** at scale | Distributed; high write throughput; multi-node; strong consistency and transactions. Built for write-heavy and scan across many keys.                                                     | **B**/**C** (read-mostly single-node): adds operational and latency cost vs ETS/SQLite when you don’t need distribution.                               | ~100k per node                     |
| **Object store (S3/GCS)** | **C**, bulk write                   | Authoritative for blobs; each node downloads to local file cache. No shared KV.                                                                                                           | **F**: append/overwrite only; not for fine-grained read-modify-write.                                                                                  | ~3.5k requests/sec per prefix      |
| **File (local)**          | **C**, bulk write                   | Cache of object store or authoritative when local-only.                                                                                                                                   | Same as object store for semantics; file is the cache layer when object store exists.                                                                  | ~10k (local I/O)                   |
| **DuckDB** (columnar)     | **E**, **C**, **D**                 | In-process; columnar; Parquet; excellent for scan/aggregate; ACID; zero-config. [DuckDB](https://duckdb.org/); [benchmarks](https://duckdb.org/2025/10/09/benchmark-results-14-lts.html). | **B**/**F**: point reads by key and read-modify-write are not primary strengths; row stores win.                                                       | ~100k–1M rows/s scan (single node) |

---

## RecGPT (this repo): YCSB by artifact

Applying the technique to RecGPT pipeline and serve:

| Mode                   | Store / artifact                                      | YCSB         | Access pattern                                                                      | Object store                          | File (cache)                     | ETS                         | SQLite                      | Throughput (est.)                          |
| ---------------------- | ----------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------- | --------------------------- | --------------------------- | ------------------------------------------ |
| **Zero-shot**          | Fixture (trie, token_id_list)                         | **C**        | Bulk read at load; read-only lookups per Predict.                                   | Authoritative; node downloads to file | Local cache or authoritative     | Hot path after load         | DB-backed catalog, restarts | ~500k lookup; ~3.5k load from object store |
| **Zero-shot**          | Checkpoint (params)                                   | **C**        | Bulk read at load; read-only forward pass.                                          | Param blobs; each node loads          | Cache .npy + manifest            | In-memory (Nx) after load   | —                           | ~3.5k load; in-memory after load           |
| **Zero-shot**          | Eval data (test/cold)                                 | **C**        | Stream or random read; no writes.                                                   | Large eval sets                       | JSON stream (cache or local)     | If fully loaded             | Streamed eval               | ~10k file stream; ~3.5k object store       |
| **Pretraining**        | Train data                                            | **C**        | Stream batches; read-only.                                                          | Sharded train blobs                   | Stream (cache or local)          | If loaded for random access | Durable train set           | ~10k file; ~3.5k object store              |
| **Pretraining**        | Fixture                                               | **C**        | Read for encoding; no writes.                                                       | Same as zero-shot                     | Same as zero-shot                | Hot path                    | Durable catalog             | ~500k lookup; ~10k file load               |
| **Pretraining**        | Params (in-memory)                                    | **F**        | Read → forward/backward → update params.                                            | —                                     | —                                | Single-node (Nx/ETS)        | —                           | ~500k in-process                           |
| **Pretraining**        | Checkpoint (on-disk export)                           | **A-like**   | Bulk write at save; no read-from-disk during training.                              | Upload blobs; authoritative           | manifest + .npy (cache or local) | —                           | Manifest/metadata only      | ~10k file write; ~3.5k object store upload |
| **Sniper / Butterfly** | Extended catalog (item_id → condition_id, outcome_id) | **B**, **C** | Point read by item_id; occasional bulk upsert when catalog syncs                    | —                                     | items.json or DB                 | ETS hot path                | SQLite, Ecto                | ~500k lookup; ~10k sync                    |
| **Sniper / Butterfly** | Implication graph (condition_id → successors)         | **C**, **E** | Point read by condition_id (get successors); optional full-edge scan for validation | —                                     | implication_graph.json           | ETS after load              | —                           | ~500k lookup; ~10k load                    |

**Summary:** Data and fixture are **Workload C** (read-only or stream); in-memory params are **F** (read-modify-write). Use **object store** + **file (cache)** + **ETS** or **SQLite** as in [13 Infrastructure](13_infrastructure_serving.md); batch object-store ops (one GET/PUT per artifact) to keep request count low.

---

## Item relationship mapper: YCSB classification

The **item relationship mapper** supports Scout→butterfly: it maps `item_id` → `(condition_id, outcome_id, market_id)` and stores the implication graph (condition A ⇒ condition B). See [60 Rope bridge §5.4](60_rope_bridge_market_analytics_plan.md#54-cross-market-implication-graph-technical-design).

### Access patterns

| Component                                           | Build phase                                                            | Serve / inference phase                                                   | Dominant ops                   |
| --------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------ |
| **Catalog** (item_id → condition_id, outcome_id)    | Bulk upsert when syncing from Polymarket                               | Point read by item_id for each Scout output and graph build               | Point read by key              |
| **Implication graph** (condition_id → [successors]) | One-off: stream sequences, accumulate edges in memory, bulk write JSON | Point read by condition_id (get successors); optional scan for validation | Point read; scan if validating |

### YCSB assignment

| Store                 | YCSB                                     | Rationale                                                                                                               |
| --------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Extended catalog**  | **B** (read mostly) or **C** (read only) | Point reads by item_id dominate. Occasional bulk refresh when catalog syncs. If never updated after init → **C**.       |
| **Implication graph** | **C** primary; **E** if validation scans | Primary use: adjacency lookup by condition_id (point read). Optional: iterate all edges for backtest/validation (scan). |

### Store fit

| Store                 | Catalog                             | Implication graph                                                          |
| --------------------- | ----------------------------------- | -------------------------------------------------------------------------- |
| **SQLite (Ecto)**     | ✓ Strong (B, C)                     | ✓ For persistence; load to ETS for hot path                                |
| **ETS**               | ✓ Hot path after load               | ✓ Hot path after load; adjacency O(1) lookup                               |
| **File**              | ✓ items.json or mapping JSON        | ✓ implication_graph.json; bulk load at startup                             |
| **Redis**             | ✓ If catalog shared across nodes    | ✓ Same as ETS for single-node                                              |
| **DuckDB** (columnar) | — Point reads weaker than row store | ✓ For validation/backtest scans (E); bulk load edges, run analytic queries |

**Recommendation:** Extend catalog in SQLite or file (item_id → condition_id, outcome_id). Build implication graph to JSON; load at startup into ETS for O(1) successor lookup. Same pattern as fixture: file/DB authoritative, ETS hot path. Use **DuckDB** if you need analytical scans over edges (e.g. backtest validation, aggregate stats) rather than point lookups.

---

## See also

- [13 Infrastructure and serving](13_infrastructure_serving.md) — Catalog storage, object-store options, run serve.
- [30 waffle_ecto usage](30_waffle_ecto_usage.md) — Blob/artifact storage with Ecto and optional S3/GCS.
- [YCSB](https://github.com/brianfrankcooper/YCSB) — Yahoo! Cloud Serving Benchmark (workload definitions).
