# Eval data shapes

Canonical JSON shapes for eval inputs so tests and tools can generate or consume data without a real dataset. All IDs are 0-based and in `0..num_items-1` unless noted.

---

## test_sequences.json

Next-item prediction test set: one case per sequence; last item is the target. Used by `Eval.load_test_cases/1`.

| Key          | Type | Description                                                               |
| ------------ | ---- | ------------------------------------------------------------------------- |
| `num_items`  | int  | Catalog size.                                                             |
| `test_cases` | list | Each element: `context` (list of item_ids), `next_item` (single item_id). |

`context` is typically length 1–64; `next_item` is the ground-truth next item. Eval uses `context` as input and checks whether `next_item` appears in the model’s top-k.

**Example:** See [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) and [08 Pipeline reference](08_pipeline_reference.md).

---

## cold_test_sequences.json

Same shape as `test_sequences.json`. Test cases where `next_item` is a cold item (≤ K sessions in train). **Required** for `mix recgpt.eval`; produced by Fetch.

---

## items.json

Item catalog for fixture building: id and title per item.

| Key         | Type | Description                                          |
| ----------- | ---- | ---------------------------------------------------- |
| `num_items` | int  | Catalog size; should match length of `items`.        |
| `items`     | list | Each element: `id` (int, 0-based), `title` (string). |

IDs should be unique and contiguous 0 .. num_items-1.

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

## Synthetic generators (test/support)

`RecGPT.EvalFixtures` in tests:

- **test_cases** — `generate_test_cases(num_items, n_cases, opts)` → list of `%{"context" => [...], "next_item" => id}`.
- **test_sequences payload** — `generate_test_sequences_json(num_items, n_cases, opts)` → map suitable for `Jason.encode!/1` and `Eval.load_test_cases(path)` after writing to a file.

Use these for property and integration tests without a real dataset.

---

## See also

- [05 Evaluation and testing](05_evaluation_and_testing.md) — Eval commands and null hypothesis.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Artifact layout and cold split.
- [08 Pipeline reference](08_pipeline_reference.md) — File layout and pipeline.
