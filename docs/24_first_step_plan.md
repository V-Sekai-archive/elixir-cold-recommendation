# First step plan (rope bridge)

This doc spells out the **first step** to close the minimal loop: use **Steam as known-good test vectors** to establish a baseline (Hit@k, MRR). **Eval and predict run in Elixir** (RecGPT.Serve + RecGPT.Eval); data prep, gRPC, and SQLite are Elixir. See [25 MVP guard rails](25_mvp_guard_rails.md) for tombstones.

---

## Why Steam first

The **Steam dataset** (RecGPT_dataset) is a **known-good dataset** with standard train/test splits, item text, embeddings, and test sequences. We use it **as test vectors** — not as the production catalog — to:

1. **Validate the recommender** (Elixir RecGPT) on a reproducible baseline (Hit@k, MRR).
2. **Confirm the pipeline** (data → Elixir eval) works before other catalogs or integrations.
3. **Compare** zero-shot vs trained and reject the random baseline.

Once Steam eval gives a baseline we trust, the **same recommendation pipeline** (Elixir Serve + gRPC) can be used with other catalogs; next steps will be replanned later.

---

## First step (concrete)

**Goal:** Run eval on the Steam test vectors and get Hit@k and MRR using the **Elixir** RecGPT stack.

### One-shot (Elixir eval)

1. **Get Steam data** (Elixir fetch produces items, sequences, and test_sequences.json.)
   - Option A: `mix recgpt.fetch_steam data/steam` (from Hugging Face).
   - Option B: Use a local clone of [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) (e.g. `path/to/RecGPT_dataset/test/steam`) and point paths below there.

2. **Run eval in Elixir.** From the repo root, with `data/steam` containing `items.json`, `fixture.json`, `test_sequences.json`, and a checkpoint export at `data/recgpt_ckpt_export`:

   ```bash
   mix recgpt.first_step
   ```

   This runs: fetch (if not skipped) → build_fixture (with `--embeddings-npy` and `--vae-ckpt` for parity) → **eval in Elixir** (RecGPT.Serve.load_state + RecGPT.Eval.evaluate). Defaults: `--steam-dir data/steam`, `--ckpt data/recgpt_ckpt_export`. Use `--skip-fetch`, `--skip-build` if data/fixture already exist.

   **Prerequisite:** RecGPT checkpoint export (manifest + .npy) must exist. Create it with:

   ```bash
   mix recgpt.fetch_ckpt
   mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
   ```

   For fixture build to match the released tokenizer, use the dataset `item_text_embeddings.npy` and VAE checkpoint: `mix recgpt.build_fixture --embeddings-npy data/steam/item_text_embeddings.npy --vae-ckpt path/to/vae_len4_fsq88865_ep90.pt`. First step does this when `item_text_embeddings.npy` is present and `--vae-ckpt` is set (or `RECGPT_VAE_CKPT`).

3. **Outcome (Elixir)**
   - You get **Hit@k** and **MRR** (or NDCG from the Python script) on the known-good Steam test set. That is the **baseline**. Document or note the numbers for later comparison.

---

### Manual steps (Python)

1. **Get Steam data** — Same as above (fetch_steam or local clone).

2. **Prepare data** — Steam dir (e.g. `data/steam`) should have `items.json`, `item_text_embeddings.npy` (for build_fixture with `--embeddings-npy`), `test_sequences.json`, and a built `fixture.json`. Checkpoint export dir must have `manifest.json` and .npy tensors.

3. **Run eval (Elixir)** — `mix recgpt.eval --data-dir data/steam --ckpt data/recgpt_ckpt_export` (uses fixture.json and test_sequences.json under data-dir). Optional: `--fixture`, `--test`, `--cold`, `--top-k`, `--progress N`.

4. **Outcome** — Hit@1, Hit@5, Hit@10, MRR, rejects_null as the baseline.

---

## See also

- [25 MVP guard rails](25_mvp_guard_rails.md) — tombstones: no multi-rank SPMD / sharding until the minimal loop is closed.
- [28 Thirdparty vs Elixir parity](28_thirdparty_vs_elixir_parity.md) — dataset .npy + VAE for parity with the released model.
- [embedding_vs_eval](embedding_vs_eval.md) — generating embeddings vs testing recommendation performance.
- [06 Evaluation and testing](06_evaluation_and_testing.md) — metrics, null hypothesis.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, eval commands.
