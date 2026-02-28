# Pipeline reference

Single reference for the recommended pipeline: **Fetch → build_fixture → pretrain → eval**. Commands, options, and file layout.

See also: [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) (concepts and artifact table), [00 RecGPT library](00_recgpt_library.md) (modules).

---

## Pipeline overview

**Order:** 1 → 2 → 3 → 4. Both standard test and cold-test files are **required** for eval. For the diagram, see [Documentation index](README.md#pipeline-overview).

---

## Step 1: Generate data

**Goal:** Produce items and train/test/cold sequences.

**Command:** `mix recgpt.clickstream` (or `mix recgpt.clickstream data/clickstream`).

**Programmatic:** `RecGPT.Clickstream.Fetch.run("data/clickstream", max_train_sessions_for_cold: 2)`.

**Outputs (under the data dir):**

| File                        | Description                                                                       |
| --------------------------- | --------------------------------------------------------------------------------- |
| `items.json`                | Catalog: `{"items": [{"id", "title"}], "num_items"}`.                             |
| `train_sequences.json`      | `{"sequences": [[id, ...], ...], "num_items"}` — 80% of sessions.                 |
| `test_sequences.json`       | `{"test_cases": [{"context", "next_item"}], "num_items"}` — 20% last-item-out.    |
| `cold_test_sequences.json`  | Same shape as test; only cases where `next_item` is cold (≤ K sessions in train). |
| `cold_train_sequences.json` | Train sequences that contain at least one cold item.                              |

Cold threshold K defaults to 2; override with `:max_train_sessions_for_cold` in `run/2` opts.

---

## Step 2: Build fixture

**Goal:** Turn `items.json` into `fixture.json` (num_items, token_id_list) via Embedding + FSQ.

**Command:**

```bash
mix recgpt.build_fixture --items data/clickstream/items.json --out data/clickstream/fixture.json --ckpt data/recgpt_ckpt_export
```

If the checkpoint does not contain FSQ params (e.g. `project_in/kernel` or `fsq.project_in.weight`), add `--fsq path/to/fsq_export`.

**Output:** `fixture.json` with `num_items` and `token_id_list`. Same format as expected by Serve and the eval task.

---

## Step 3: Pretrain

**Goal:** Train on `train_sequences.json` using fixture and checkpoint; write updated params to an export dir.

**Command:**

```bash
mix recgpt.pretrain --ckpt data/recgpt_ckpt_export --fixture data/clickstream/fixture.json --train data/clickstream/train_sequences.json --items data/clickstream/items.json --out data/ckpt_after_pretrain --iterations 100 --batch-size 8 --log 50
```

Optional: `--embeddings path/to/embeddings.nx` for precomputed embeddings.

**Output:** New export dir (e.g. `data/ckpt_after_pretrain`) with `manifest.json` and `.npy` files. Use this dir as `--ckpt` for eval and serve.

---

## Step 4: Eval

**Goal:** Run next-item evaluation on both standard test and cold-test sets; print Hit@k, MRR, and null rejection.

**Command:**

```bash
mix recgpt.eval --fixture data/clickstream/fixture.json --ckpt data/ckpt_after_pretrain --test data/clickstream/test_sequences.json --cold-test data/clickstream/cold_test_sequences.json
```

**Requirements:** Fixture and checkpoint must exist and load. **Both** `--test` and `--cold-test` are required; if `cold_test_sequences.json` is missing, the task fails and prompts you to run Fetch first.

**Output:** Two blocks of metrics: “Evaluation (standard test set)” and “Cold test”.

---

## Optional: Serve

After pretrain (and optionally eval):

```bash
mix recgpt.serve --fixture data/clickstream/fixture.json --ckpt data/ckpt_after_pretrain [--grpc-port 50051]
```

gRPC only: **recgpt.v1.PredictionService/Predict**. Contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). [13](13_grpc_rest_api.md).

---

## Checkpoint setup (before pipeline)

If you do not have an export dir yet:

```bash
mix recgpt.fetch_ckpt
mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
```

Use `data/recgpt_ckpt_export` as `--ckpt` for build_fixture and pretrain.

---

## File layout summary

```
data/
├── recgpt_ckpt_export/          # From fetch_ckpt + export_ckpt
│   ├── manifest.json
│   └── *.npy
├── recgpt_layer_3_weight.pt     # Optional; from fetch_ckpt
├── clickstream/
│   ├── items.json
│   ├── train_sequences.json
│   ├── test_sequences.json
│   ├── cold_test_sequences.json
│   ├── cold_train_sequences.json
│   └── fixture.json             # From build_fixture
└── ckpt_after_pretrain/         # From pretrain --out
    ├── manifest.json
    └── *.npy
```

---

## Environment variables

| Variable             | Used by     | Purpose                         |
| -------------------- | ----------- | ------------------------------- |
| `RECGPT_FIXTURE`     | eval, serve | Override fixture path.          |
| `RECGPT_CKPT_EXPORT` | eval, serve | Override checkpoint export dir. |

Command-line options override these.

---

## See also

- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Splits and artifact semantics.
- [05 Evaluation and testing](05_evaluation_and_testing.md) — Eval metrics and null hypothesis.
- [06 Eval data shapes](06_eval_data_shapes.md) — JSON shapes.
- [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto) — gRPC API contract (Serve).
- [13 gRPC REST API](13_grpc_rest_api.md) — gRPC + REST design; proto in `priv/proto/recgpt/v1/`.
- [00 RecGPT library](00_recgpt_library.md) — Module reference.
