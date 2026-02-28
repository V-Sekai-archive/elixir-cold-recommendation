# Pipeline overview

Sub-proposal of the [documentation index](README.md). Pipeline order and Step 1 (generate data). Commands and layout: [03 Pipeline steps](03_pipeline_steps.md).

---

## Problem or limitation

We need one reproducible path from raw data to trained model and metrics. Without a single pipeline specification (order, commands, options, file layout), users and automation invent ad-hoc sequences and results are not comparable.

---

## Proposed improvement

Define the **pipeline** as four steps with commands, options, and outputs. Both standard test and cold-test files are required for eval. Diagram: [Documentation index](README.md#pipeline-overview). Concepts: [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md); modules: [04 RecGPT library](04_recgpt_library.md).

---

## Pipeline overview

**Order:** 1 → 2 → 3 → 4. Both standard test and cold-test files are **required** for eval. For the diagram, see [Documentation index](README.md#pipeline-overview).

---

## Step 1: Generate data

**Goal:** Produce items and train/test/cold sequences.

**Command:** `mix recgpt.fetch_steam data/steam` (or another output dir).

**Programmatic:** `RecGPT.Steam.Fetch.run("data/steam")`.

**Outputs (under the data dir):**

| File                        | Description                                                                    |
| --------------------------- | ------------------------------------------------------------------------------ |
| `items.json`                | Catalog: `{"items": [{"id", "title"}], "num_items"}`.                          |
| `train_sequences.json`      | `{"sequences": [[id, ...], ...], "num_items"}` — 80% of sessions.              |
| `test_sequences.json`       | `{"test_cases": [{"context", "next_item"}], "num_items"}` — 20% last-item-out. |
| `cold_test_sequences.json`  | Same shape as test; only cases where `next_item` is cold.                      |
| `cold_train_sequences.json` | Train sequences that contain at least one cold item.                           |

Cold files are produced by Steam Fetch from the dataset.

---

## See also

- [03 Pipeline steps](03_pipeline_steps.md) — Steps 2–4, serve, layout, env.
