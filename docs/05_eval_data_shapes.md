git# Proposal: Eval data shapes

Sub-proposal of the [documentation index](README.md). Canonical JSON shapes for pipeline and eval artifacts.

---

## Problem or limitation

Tests and tools need canonical JSON shapes for test_sequences, items, fixture, train_sequences, and cold files so they can generate or consume data without a real dataset. Ad-hoc shapes cause parse errors and incompatibility across steps.

---

## Proposed improvement

Define **one shape per artifact**: keys, types, and semantics. All IDs are 0-based and in `0..num_items-1` unless noted. Implementations (e.g. `Eval.load_test_cases/1`, `FixtureBuild`) follow these shapes.

---

## test_sequences.json

Next-item prediction test set: one case per sequence; last item is the target. Used by `Eval.load_test_cases/1`.

| Key          | Type | Description                                                               |
| ------------ | ---- | ------------------------------------------------------------------------- |
| `num_items`  | int  | Catalog size.                                                             |
| `test_cases` | list | Each element: `context` (list of item_ids), `next_item` (single item_id). |

`context` is typically length 1–64; `next_item` is the ground-truth next item. Eval uses `context` as input and checks whether `next_item` appears in the model's top-k.

**Example:** See [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) and [02 Pipeline overview](02_pipeline_overview.md).

---

## cold_test_sequences.json

Same shape as `test_sequences.json`. Test cases where `next_item` is a cold item (≤ K sessions in train). **Required** for `mix recgpt.eval`; produced by Fetch.

---

## items.json

Item catalog for fixture building and for Serve display names (id and title per item). This file is the **SSD-stable** canonical store for catalogue item data.

| Key         | Type | Description                                          |
| ----------- | ---- | ---------------------------------------------------- |
| `num_items` | int  | Catalog size; should match length of `items`.        |
| `items`     | list | Each element: `id` (int, 0-based), `title` (string). |

IDs should be unique and contiguous 0 .. num_items-1.

**SSD-stable storage:** When writing catalog (e.g. from a Mix task or future API), use atomic replace so the visible file is never half-written: write to a temporary file in the same directory (e.g. `items.json.tmp`), optionally `File.sync/1` on the temp file, then `File.rename/2` to the final path. On most filesystems rename is atomic. Serve and pipeline read the file once at load time; no in-place updates.

---

## fixture.json

Tokenized catalog for Serve and Eval: one 4-token FSQ sequence per item.

| Key             | Type          | Description                                          |
| --------------- | ------------- | ---------------------------------------------------- |
| `num_items`     | int           | Catalog size.                                        |
| `token_id_list` | list of lists | One list of 4 integers (FSQ vocab indices) per item. |

Values are typically in 0 .. 15359. Built from items + Embedding + FSQ via `mix recgpt.build_fixture` or `RecGPT.FixtureBuild`.

---

## train_sequences.json

Full sequences for pretraining (no last-item-out). Same catalog as test; sessions are split into train vs test.

| Key         | Type          | Description                                              |
| ----------- | ------------- | -------------------------------------------------------- |
| `num_items` | int           | Catalog size.                                            |
| `sequences` | list of lists | Each inner list is a full sequence of item_ids in order. |

Used by `RecGPT.Training.build_train_batch/4` with token_id_list and item embeddings. Train and test must be disjoint (e.g., 80% sessions → train, 20% → test last-item-out). See [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md).

---

## cold_train_sequences.json

Same shape as `train_sequences.json`. Train sequences that contain at least one cold item. Produced by Fetch; optional for training.

---

## Sub-proposals

- **test_sequences.json** (above) — Test set shape for `Eval.load_test_cases/1`.
- **cold_test_sequences.json** (above) — Same shape; cold items only.
- **items.json** (above) — Catalog for fixture building.
- **fixture.json** (above) — Tokenized catalog for Serve and Eval.
- **train_sequences.json** (above) — Pretraining sequences.
- **cold_train_sequences.json** (above) — Train sequences containing cold items.

---

## See also

- [06 Evaluation and testing](06_evaluation_and_testing.md) — Eval commands and null hypothesis.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Artifact layout and cold split.
- [02 Pipeline overview](02_pipeline_overview.md) — File layout and pipeline.
