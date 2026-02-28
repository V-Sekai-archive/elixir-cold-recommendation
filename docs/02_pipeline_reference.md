# Proposal: Pipeline reference

Sub-proposal of the [documentation index](README.md). Single reference for the recommended pipeline: **Fetch → build_fixture → pretrain → eval**.

---

## Problem or limitation

We need one reproducible path from raw data to trained model and metrics. Without a single pipeline specification (order, commands, options, file layout), users and automation invent ad-hoc sequences and results are not comparable.

---

## Proposed improvement

Define the **pipeline** as four steps with commands, options, and outputs. Both standard test and cold-test files are required for eval. Diagram: [Documentation index](README.md#pipeline-overview). Concepts: [06 Steam splits and pretraining](06_steam_splits_and_pretraining.md); modules: [03 RecGPT library](03_recgpt_library.md).

---

## Pipeline overview

**Order:** 1 → 2 → 3 → 4. Both standard test and cold-test files are **required** for eval. For the diagram, see [Documentation index](README.md#pipeline-overview).

---

## Step 1: Generate data

**Goal:** Produce items and train/test/cold sequences.

**Command:** `mix recgpt.fetch_steam data/steam` (or another output dir).

**Programmatic:** `RecGPT.Steam.Fetch.run("data/steam")`.

**Outputs (under the data dir):**

| File                        | Description                                                                       |
| --------------------------- | --------------------------------------------------------------------------------- |
| `items.json`                | Catalog: `{"items": [{"id", "title"}], "num_items"}`.                             |
| `train_sequences.json`      | `{"sequences": [[id, ...], ...], "num_items"}` — 80% of sessions.                 |
| `test_sequences.json`       | `{"test_cases": [{"context", "next_item"}], "num_items"}` — 20% last-item-out.    |
| `cold_test_sequences.json`  | Same shape as test; only cases where `next_item` is cold (≤ K sessions in train). |
| `cold_train_sequences.json` | Train sequences that contain at least one cold item.                              |

Cold files are produced by Steam Fetch from the dataset.

---

## Step 2: Build fixture

**Goal:** Turn `items.json` into `fixture.json` (num_items, token_id_list) via Embedding + FSQ.

**Command:**

```bash
mix recgpt.build_fixture --items data/steam/items.json --out data/steam/fixture.json --ckpt data/recgpt_ckpt_export
```

**Output:** `fixture.json` with `num_items` and `token_id_list`. Same format as expected by Serve and the eval task.

---

## Step 3: Pretrain

**Goal:** Train on `train_sequences.json` using fixture and checkpoint; write updated params to an export dir.

**Command:**

```bash
mix recgpt.pretrain --ckpt data/recgpt_ckpt_export --fixture data/steam/fixture.json --train data/steam/train_sequences.json --items data/steam/items.json --out data/ckpt_after_pretrain --iterations 100 --batch-size 8 --log 50
```

**Output:** New export dir (e.g., `data/ckpt_after_pretrain`) with `manifest.json` and `.npy` files. Use this dir as `--ckpt` for eval and serve.

---

## Step 4: Eval

**Goal:** Run next-item evaluation on both standard test and cold-test sets; print Hit@k, MRR, and null rejection.

**Command:**

```bash
mix recgpt.eval --fixture data/steam/fixture.json --ckpt data/ckpt_after_pretrain --test data/steam/test_sequences.json --cold-test data/steam/cold_test_sequences.json
```

**Requirements:** Fixture and checkpoint must exist and load. **Both** `--test` and `--cold-test` are required; if `cold_test_sequences.json` is missing, the task fails and prompts you to run Fetch first.

**Output:** Two blocks of metrics: "Evaluation (standard test set)" and "Cold test".

---

## Optional: Serve

After pretrain (and optionally eval):

```bash
mix recgpt.serve --fixture data/steam/fixture.json --ckpt data/ckpt_after_pretrain [--grpc-port 50051]
```

gRPC only: **recgpt.v1.PredictionService/Predict**. Contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). See [01 gRPC API](01_grpc_api.md).

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
├── steam/
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

## Sub-proposals

- [Step 1: Generate data](#step-1-generate-data) — Fetch; outputs.
- [Step 2: Build fixture](#step-2-build-fixture) — build_fixture; fixture.json.
- [Step 3: Pretrain](#step-3-pretrain) — pretrain; updated checkpoint.
- [Step 4: Eval](#step-4-eval) — eval; metrics.

---

## See also

- [06 Steam splits and pretraining](06_steam_splits_and_pretraining.md) — Splits and artifact semantics.
- [05 Evaluation and testing](05_evaluation_and_testing.md) — Eval metrics and null hypothesis.
- [04 Eval data shapes](04_eval_data_shapes.md) — JSON shapes.
- [01 gRPC API](01_grpc_api.md) — gRPC contract and serve.
- [03 RecGPT library](03_recgpt_library.md) — Module reference.
