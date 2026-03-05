# ETNF Database Design

Catalog and sequence storage in Essential Tuple Normal Form (ETNF) for RecGPT.

## Overview

ETNF (Darwen, Date, Fagin 2012) lies between 4NF and 5NF and eliminates redundant tuples. Relations are in BCNF and every explicitly declared join dependency has a superkey component. Applied here: each relation has a declared superkey; Sync deduplicates by key before insert.

## Schema

| Relation | Superkey | FDs |
|----------|----------|-----|
| items | (item_id) | item_id → title |
| item_embedding_texts | (item_id) | item_id → embedding_text |
| item_embeddings | (item_id) | item_id → embedding |
| item_tokens | (item_id) | item_id → (t0, t1, t2, t3) |
| train_sequence_rows | (seq_id, pos) | (seq_id, pos) → item_id |
| cold_train_sequence_rows | (seq_id, pos) | (seq_id, pos) → item_id |
| test_cases | (case_id) | case_id → next_item |
| test_context | (case_id, pos) | (case_id, pos) → item_id |
| cold_test_cases | (case_id) | case_id → next_item |
| cold_test_context | (case_id, pos) | (case_id, pos) → item_id |

## Enforcement

- **Unique indexes**: `train_sequence_rows(seq_id, pos)`, `test_context(case_id, pos)`, etc.
- **Deduplication**: `RecGPT.Catalog.Sync` calls `Enum.uniq_by` on (seq_id, pos) or (case_id, pos) before insert.
- **NOT NULL**: All key and dependent attributes are non-null in migrations.

## Convert → DB Flow (no JSON after ingestion)

With `--sync-to-db`, `mix recgpt.convert_trajectories` writes:
- Items and sequences to SQLite only (no JSON files)
- Use `--items db`, `--train db`, `--test db` for downstream steps

Requires `RECGPT_SQLITE_PATH`. Run `mix ecto.migrate` once.

```bash
mix recgpt.convert_trajectories --from thirdparty/KuaiRand-Pure --out data/kuairand --format kuairand --sync-to-db
mix recgpt.build_fixture --items db --out data/kuairand/fixture.json
mix recgpt.pretrain --train db --items db --fixture data/kuairand/fixture.json --out data/kuairand/ckpt
mix recgpt.eval --fixture data/kuairand/fixture.json --ckpt data/kuairand/ckpt --test db
```

**item_embedding_texts:** Text used as embedding input. When present, use it; when absent, fall back to `items.title`. Polymarket: canonical JSON-LD (RFC 8785 JCS); Steam/KuaiRand: typically nil (embedding uses title). See [92 Polymarket semantic source](92_polymarket_semantic_source.md).

## See also

- [13 Infrastructure and serving](13_infrastructure_serving.md)
- [30 waffle_ecto usage](30_waffle_ecto_usage.md)
