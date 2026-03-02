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

### Prerequisites

1. **RecGPT checkpoint export** (manifest.json + .npy):

   ```bash
   mix recgpt.fetch_ckpt
   mix recgpt.export_ckpt --from-pt thirdparty/checkpoints/recgpt/recgpt_layer_3_weight.pt --out thirdparty/checkpoints/recgpt
   ```

2. **VAE checkpoint** for FSQ parity (`vae_len4_fsq88865_ep90.pt`):

   ```bash
   mix recgpt.fetch_vae_ckpt
   ```

3. **Canonical item texts** (byte-exact with RecGPT official; run once after fetch):

   ```bash
   mix ecto.migrate
   uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --verify
   ```

   `build_fixture` uses these by default; semantic IDs match the released model.

### One-shot (Elixir eval)

1. **Get Steam data** (Elixir fetch produces items, sequences, and test_sequences.json.)

   ```bash
   mix recgpt.fetch_steam data/steam
   ```

   Or use a local clone of [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) (e.g. `path/to/RecGPT_dataset/test/steam`).

2. **Run first step** (fetch → build_fixture → eval in Elixir):

   ```bash
   mix recgpt.first_step
   ```

   Defaults: `--steam-dir data/steam`, `--ckpt thirdparty/checkpoints/recgpt`, `--vae-ckpt` from `RECGPT_VAE_CKPT`. Use `--skip-fetch`, `--skip-build` if data/fixture already exist.

   Build uses canonical texts from SQLite + Bumblebee encoder + VAE FSQ → token_id_list matches released checkpoint.

3. **Outcome (Elixir)** — Hit@1, Hit@5, Hit@10, MRR, rejects_null on the known-good Steam test set. Document or note the numbers for later comparison.

---

### Manual steps

1. **Get Steam data** — `mix recgpt.fetch_steam data/steam` (or local clone).
2. **Prepare canonical texts** — Run `dump_canonical_to_sqlite.py` once (see Prerequisites).
3. **Build fixture** — `mix recgpt.build_fixture --items data/steam/items.json --out data/steam/fixture.json --vae-ckpt path/to/vae_len4_fsq88865_ep90.pt --limit 10000`
4. **Run eval** — `mix recgpt.eval --data-dir data/steam --ckpt thirdparty/checkpoints/recgpt` (uses fixture.json and test_sequences.json). Optional: `--fixture`, `--test`, `--cold`, `--top-k`, `--progress N`.
5. **Outcome** — Hit@1, Hit@5, Hit@10, MRR, rejects_null as the baseline.

---

## Continue plan (now that Steam semantic IDs match)

With canonical texts and 100% FSQ agreement, we can run the full first step end-to-end. Next concrete actions:

1. **Run first step** — `mix recgpt.first_step` (or manual steps) and record baseline Hit@k, MRR.
2. **Compare zero-shot vs trained** — Eval with pretrained checkpoint (zero-shot), then pretrain on `train_sequences.json`, then eval with fine-tuned checkpoint. Reject the random baseline.
3. **Document metrics** — Note Hit@1, Hit@5, Hit@10, MRR for reproducibility.
4. **Next steps** — Once baseline is established, replan for other catalogs or integrations.
5. **Future: custom catalogues and pretraining** — Use JSON-LD (e.g. [jsonld-ex](https://github.com/rdf-elixir/jsonld-ex)) for item metadata: author/validate as JSON-LD with a shared @context, then derive the encoder input string (fixed key order) so the same RecGPT pipeline applies. Steam stays byte-exact for parity; new catalogs and pretraining can be JSON-LD-first.

---

## See also

- [25 MVP guard rails](25_mvp_guard_rails.md) — tombstones: no multi-rank SPMD / sharding until the minimal loop is closed.
- [28 Thirdparty vs Elixir parity](28_thirdparty_vs_elixir_parity.md) — dataset .npy + VAE for parity with the released model.
- [embedding_vs_eval](embedding_vs_eval.md) — generating embeddings vs testing recommendation performance.
- [06 Evaluation and testing](06_evaluation_and_testing.md) — metrics, null hypothesis.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, eval commands.
