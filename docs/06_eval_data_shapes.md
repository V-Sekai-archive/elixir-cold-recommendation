# Eval data shapes (for synthetic generators)

Canonical shapes for eval inputs so tests and tools can generate synthetic data without real datasets.

---

## 1. test_sequences.json (Eval.load_test_cases/1)

Next-item prediction test set: one case per sequence, last item is the target.

```json
{
  "num_items": 188,
  "test_cases": [
    { "context": [94, 34, 102], "next_item": 114 },
    { "context": [99, 52], "next_item": 148 }
  ]
}
```

| Key           | Type   | Description |
|---------------|--------|-------------|
| `num_items`   | int    | Catalog size (item IDs in 0 .. num_items-1). |
| `test_cases`  | list   | Each element: `context` (list of item_ids), `next_item` (single item_id). |
| `context`     | [int]  | Ordered sequence of item IDs (length 1..64 typical). |
| `next_item`   | int    | Ground-truth next item ID (in 0 .. num_items-1). |

- All IDs in `context` and `next_item` must be in `0 .. num_items-1`.
- Eval uses `context` as input and checks if `next_item` appears in the model’s top-k.

---

## 2. items.json (catalog for fixture building)

Item catalog: id + title per item. Used to build embeddings and then fixture.

```json
{
  "num_items": 188,
  "items": [
    { "id": 0, "title": "category 28 product 2 colour 1" },
    { "id": 1, "title": "category 4 product P59 colour 2" }
  ]
}
```

| Key        | Type   | Description |
|------------|--------|-------------|
| `num_items`| int    | Same as length of `items`. |
| `items`    | list   | Each element: `id` (int, 0-based), `title` (string). |

- IDs should be unique and contiguous 0 .. num_items-1 for simplicity.

---

## 3. fixture.json (Serve / Eval state)

Tokenized catalog for the server: one 4-token FSQ sequence per item.

```json
{
  "num_items": 188,
  "token_id_list": [
    [120, 45, 3, 0],
    [98, 12, 7, 1]
  ]
}
```

| Key             | Type    | Description |
|-----------------|---------|-------------|
| `num_items`     | int     | Catalog size. |
| `token_id_list` | [[int]] | List of 4-token lists (FSQ vocab indices per item). |

- Each inner list has 4 integers (FSQ encoding). Vocab size 15360 + padding; values typically in 0 .. 15359.
- Built from items + Embedding + FSQ; not usually generated purely synthetically for real eval.

---

## 4. Synthetic generators (test/support)

Use `RecGPT.EvalFixtures` in tests to generate:

- **test_cases** – `generate_test_cases(num_items, n_cases, opts)` → list of `%{"context" => [...], "next_item" => id}` with IDs in range.
- **test_sequences payload** – `generate_test_sequences_json(num_items, n_cases, opts)` → map suitable for `Jason.encode!/1` and `Eval.load_test_cases(path)` after writing to a file.

This allows property and integration tests to run without real Clickstream (or any) dataset.

---

## 5. train_sequences.json (Steam-like; for pretraining)

Full sequences for training (no last-item-out). Same catalog as test; split sessions into train vs test.

```json
{
  "num_items": 188,
  "sequences": [
    [12, 5, 99, 3],
    [0, 1, 2]
  ]
}
```

| Key         | Type     | Description |
|-------------|----------|-------------|
| `num_items` | int      | Catalog size. |
| `sequences` | [[int]]  | List of full sequences (each = list of item_ids in order). |

- Used for pretraining: build batches via `RecGPT.Training.build_train_batch/4` (with token_id_list and item embeddings).
- Train and test must be disjoint (e.g. 80% sessions → train_sequences, 20% → test_sequences last-item-out). See [07 Steam dataset splits and pretraining](07_steam_splits_and_pretraining.md).

---

## 6. cold_test_sequences.json (optional)

Same shape as test_sequences.json. Test cases where `next_item` is a cold item (unseen or rare in train). Used to measure cold-start performance.
