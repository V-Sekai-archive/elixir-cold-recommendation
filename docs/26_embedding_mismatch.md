# Embedding mismatch: our Bumblebee vs dataset item_text_embeddings.npy

Sub-proposal of the [documentation index](README.md). Describes the **embedding parity gap** and how to check or debug it.

**Scope:** This doc is only about **generating embeddings** and **checking parity** with the reference `.npy`. Testing recommendation performance (Hit@k, MRR, eval, serve) is a separate concern — see [embedding_vs_eval.md](embedding_vs_eval.md) and [06 Evaluation and testing](06_evaluation_and_testing.md).

---

## Text format pattern (what we know)

- **item_text_dict.pkl** (reference): values are **maps with a `"title"` key** (e.g. `%{"title" => "Papers, Please"}`). Inspect with:
  ```bash
  mix recgpt.inspect_item_text --steam-dir data/steam --limit 5
  ```
- **Our encoding:** We build the string as `'title': '<title>'` (single-quoted key and value, no outer braces) to match the doc’s “Python `str(dict).replace('{','').replace('}','')`” idea.
- **Empirical check:** Comparing to the dataset .npy, **`recgpt_item_text` (dict-style) gives higher cosine similarity than plain title**. With `--text-format title_only`, mean cos_sim drops (e.g. ~0.47 vs ~0.60). So the reference is likely **not** encoding the bare title; something dict-style is more likely.
- **Pipeline alignment:** Elixir uses fixed `sequence_length: 384` (sentence-transformers default for all-mpnet-base-v2), mean pooling with attention mask, and `embedding_processor: :l2_norm` so processing matches the reference.
- **Remaining gap:** Even with dict-style text we see mean cos_sim ~0.60 (large mismatch). Possible causes: (1) Bumblebee vs sentence_transformers tokenization/pooling differences, (2) subtle string differences (quotes, key order, escaping), (3) row order if the .npy was built with a different item order (the compare task prints a row-order warning when our item 0’s best match in the .npy is not row 0).

---

## Problem

Our Elixir pipeline can produce 768-d item text embeddings either with **Bumblebee** (sentence-transformers/all-mpnet-base-v2) or by **loading the dataset’s** `item_text_embeddings.npy`. When we compare our Bumblebee embeddings to that .npy **per row** (same item index), cosine similarity is well below 1.0.

- **Parity baseline (as of check):** mean cosine similarity **~0.60** (min ~0.41, max ~0.79). Verdict: **large mismatch (mean < 0.95)** between our encoder and the reference .npy.
- **Recommendation performance** (Hit@k, MRR, eval) is a separate step: it uses a fixture built from _either_ our embeddings _or_ the dataset .npy. For released-checkpoint compatibility, use the dataset .npy when building the fixture; see [embedding_vs_eval.md](embedding_vs_eval.md).

---

## Workaround: use the original dataset embeddings

We have **not** done pretraining yet, so we are not tied to our own encoder. For eval/serve with the **released checkpoint**, use the **original dataset’s** `item_text_embeddings.npy` when building the fixture so `token_id_list` matches what the model was trained on:

```bash
mix recgpt.build_fixture --embeddings-npy data/steam/item_text_embeddings.npy
```

Or with a local dataset clone (e.g. `path/to/RecGPT_dataset/test/steam`):

```bash
mix recgpt.build_fixture --items path/to/RecGPT_dataset/test/steam/items.json --embeddings-npy path/to/RecGPT_dataset/test/steam/item_text_embeddings.npy --out data/steam/fixture.json --ckpt data/recgpt_ckpt_export
```

The `.npy` must have at least as many rows as items in `items.json`; row `i` is used as the embedding for item index `i`. No Bumblebee encoding is run when `--embeddings-npy` is set.

**FSQ:** For FSQ token codes to match the reference, the same 768-d embeddings and the same FSQ params (from the checkpoint) must be used. Our `RecGPT.FSQ` loads `project_out` from the checkpoint and transposes `(192, 5)` → `(5, 192)` so the batch dot matches PyTorch’s `Linear(5, 192)`. Run `mix recgpt.compare_embeddings --steam-dir data/steam --ckpt <ckpt_dir>` to see FSQ token agreement and the first few items’ 4-token codes (ours vs dataset).

---

## How to run the check

**Inspect reference text shape** (what’s in item_text_dict.pkl and what we build):

```bash
mix recgpt.inspect_item_text --steam-dir data/steam --limit 5
```

**Compare our embeddings to the dataset .npy:**

From repo root, with Steam data present (e.g. after `mix recgpt.fetch_steam data/steam`):

```bash
mix recgpt.compare_embeddings --steam-dir data/steam --limit 500
```

Optional:

- `--limit N` — compare first N items (default 500).
- `--text-format recgpt_item_text` (default) or `title_only` — dict-style `'title': 'X'` vs plain title; use to test which format is closer to the reference.
- `--ckpt <dir>` — also report FSQ token agreement (same 4-token code per item).
- `--dump-row 0` — write our embedding for item 0 as raw float32 to `item0_elixir.raw` for Python-side diff.

The task downloads **item_text_embeddings.npy** from HuggingFace if missing (`data/steam/item_text_embeddings.npy`).

### See also: RecGPT-old (Python reference + old Elixir port)

In the **RecGPT-old** reference repo: the Python `data_processing/TextEncoder_batch.py` uses `model.encode(batch_list)` with **no** `normalize_embeddings` argument, so sentence-transformers’ **default** (L2 normalize) applies — the dataset .npy is unit norm (we see "dataset row 0: 1.0" in the compare). The **old Elixir port** in `recgpt-old/recgpt` used `embedding_processor: nil` and documented "match Python normalize_embeddings=False"; that would mismatch the actual .npy. This repo uses **`:l2_norm`** and **fixed sequence length 384** to match the reference. The Python script does not set `max_seq_length`; 384 is sentence-transformers’ default for all-mpnet-base-v2.

### See also: local RecGPT_dataset clone

If you have a local clone of the dataset ([hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset)), you can point the tasks at a test split and avoid downloading the large .npy:

- **Layout:** `test/steam`, `test/baby`, `test/games`, `test/office`, `test/washington`, `test/yelp` each contain `item_text_dict.pkl`, `item_text_embeddings.npy`, `train.pkl`, `test.pkl`, `cold_train.pkl`, `cold_test.pkl`.
- **Steam example (local clone):**
  1. Generate `items.json` and sequence JSONs from the existing .pkls:
     ```bash
     mix recgpt.fetch_steam /path/to/RecGPT_dataset/test/steam
     ```
     (Uses existing .pkl files in that dir; only downloads if a file is missing.)
  2. Inspect or compare using that dir (uses the local .npy, no 92MB download):
     ```bash
     mix recgpt.inspect_item_text --steam-dir /path/to/RecGPT_dataset/test/steam --limit 5
     mix recgpt.compare_embeddings --steam-dir /path/to/RecGPT_dataset/test/steam --limit 500
     ```

---

## What the report includes

- **First 3 encoded strings** — exact text we pass to the encoder (e.g. `'title': 'Papers, Please'`). Compare with reference `item_text_dict` / Python `str(dict).replace(...)`.
- **Row order check** — if our item 0’s best match in the .npy is not row 0, a warning is printed (suggests .npy row order ≠ our items order).
- **Scale / L2 norm** — row 0 and mean norm for ours vs dataset (both expected 1.0 if L2-normalized).
- **Cosine similarity (per item):** mean, min, max, std and a short verdict (e.g. very close ≥0.99, moderate 0.95–0.99, large mismatch <0.95).
- **FSQ token agreement** (if `--ckpt` set) — fraction of items where our FSQ 4-token code equals the one from the dataset embeddings.

---

## Likely causes of the mismatch

1. **Text format** — Our `RecGPT.Embedding.recgpt_item_text/1` builds `'title': '<escaped title>'`. The reference may use a different string (e.g. full dict, key order, or escaping). Compare the “First 3 encoded strings” with the reference `item_text_dict.pkl` / code that produced the .npy.
2. **Item/row order** — Our order comes from `items.json` (from fetch). The .npy may follow a different order (e.g. from `item_text_dict.pkl`). The row-order check warns if our row 0’s best match is not dataset row 0.
3. **Tokenization or pooling** — Same model id and mean pooling; small differences (e.g. padding, attention mask, or Bumblebee vs sentence_transformers API) can still change outputs.

---

## Debugging with a single row

To compare one item in Python:

```bash
mix recgpt.compare_embeddings --steam-dir data/steam --limit 1 --dump-row 0
```

Then in Python, load the reference .npy and our dump:

```python
import numpy as np
ref = np.load("data/steam/item_text_embeddings.npy")  # or path you use
ours = np.fromfile("item0_elixir.raw", dtype=np.float32).reshape(1, 768)
# Compare ref[0] vs ours[0], or ref[best_idx] vs ours[0] if row order differs
```

Ensure the **same input string** is used in both pipelines (e.g. print our “[0]” string and use it in Python’s encoder).

---

## Relation to parity docs

- [10 Parity by layer](10_parity_layers.md): Text → 768-d is “Done” and “Validated” for _implementation_ (model id, mean pooling, no extra L2). **Numerical parity** with the dataset .npy is not yet achieved; this doc records the gap and how to re-check.
- [09 Parity overview](09_parity_overview.md): Embeddings are “Implemented” with a note that “Elixir vs reference embeddings may differ slightly.” The compare task quantifies that difference (e.g. mean cos_sim ~0.60).

Improving the match would raise mean cosine similarity toward ≥0.95. For **testing recommendation performance** (eval, serve), use the fixture built from the dataset .npy when comparing to the released checkpoint.

---

## See also

- [embedding_vs_eval.md](embedding_vs_eval.md) — Divide: generating embeddings vs testing recommendation performance.
- [06 Evaluation and testing](06_evaluation_and_testing.md) — Hit@k, MRR, null hypothesis, eval commands.
