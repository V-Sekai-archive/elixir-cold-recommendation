# MVP guard rails (tombstones)

This repo is a **RecGPT library** for sequential recommendation. **RecGPT runs in Elixir** (Serve, Eval, Inference); Elixir provides Ecto, gRPC, SQLite, and the full pipeline. Keep the rope bridge first: one rank, one wavefront.

---

## Guard rails

Do **not** do the following until the minimal loop (catalog → fixture → recommend → eval) works end-to-end:

- **No multi-rank SPMD** in this repo — no sharded eval, no multi-rank recommend.
- **No sharded catalog or model** in this repo — single fixture, single checkpoint per process.
- **No scaling work** that assumes multiple ranks (e.g. distributed beam search across ranks).

These are **tombstones**: if you find yourself implementing them before the MVP loop is closed, stop and finish the rope bridge first.

---

## Links

- [24 First step plan](24_first_step_plan.md) — Steam test vectors (baseline).
- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, tests, performance, embedding parity.
- [embedding_vs_eval](embedding_vs_eval.md) — divide: embeddings vs recommendation performance.
