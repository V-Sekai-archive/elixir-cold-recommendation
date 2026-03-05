# Parity with the released model (dataset .npy + VAE)

Sub-proposal of the [documentation index](README.md). RecGPT inference, eval, and Predict run **entirely in Elixir** (Serve, Eval, fixture + checkpoint). To match the released model’s tokenization and recommendations, use the dataset’s item embeddings and FSQ params from the VAE checkpoint.

---

## Problem or limitation

Recommendation and eval results depend on **embeddings** and **FSQ (tokenization)**. If the fixture is built with different embeddings or FSQ params than the released model used, token_id_list and recommendations will differ. Using the **dataset** `item_text_embeddings.npy` and **FSQ from the VAE** checkpoint gives parity with the released model.

---

## Why results can differ

| Cause               | Correct (parity)                                                                                                                    | If not aligned                                                                                                                                                                                           | Effect                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Item embeddings** | Use `item_text_embeddings.npy` from the dataset.                                                                                    | Bumblebee (our encoder) gives different 768-d vectors. Our Bumblebee vs dataset .npy has **~0.60 mean cosine similarity** (see [26 Embedding mismatch](26_embedding_mismatch.md)).                       | Different 768-d vectors → different FSQ codes → different token_id_list → different recommendations. |
| **FSQ params**      | Load **FSQ from the VAE** checkpoint (`vae_len4_fsq88865_ep90.pt`). The VAE contains the FSQ `quantizer` (project_in, project_out). | Fixture build loading FSQ from the **RecGPT checkpoint export** fails: the RecGPT .pt does not include the VAE; the export has no FSQ weights, so Elixir falls back to **dummy** project_in/project_out. | Dummy FSQ → wrong token_id_list → wrong wte lookup and recommendations.                              |

So for parity: (1) use the **dataset** `item_text_embeddings.npy` when building the fixture, and (2) load **FSQ from the VAE** checkpoint when building the fixture.

---

## Parity (what to do)

To get the same token_id_list and recommendations as the released model:

1. **Use the dataset embeddings**  
   Build the fixture with the same embeddings Python uses:

   ```bash
   mix recgpt.build_fixture --embeddings-npy path/to/item_text_embeddings.npy
   ```

   (e.g. from a RecGPT_dataset clone or after fetch). Do not rely on Bumblebee for parity with the released checkpoint.

2. **Load FSQ from the VAE checkpoint**  
   The RecGPT export does not contain FSQ; the VAE .pt does. Pass the VAE path when building the fixture:

   ```bash
   mix recgpt.build_fixture --vae-ckpt path/to/vae_len4_fsq88865_ep90.pt
   ```

   Or set `RECGPT_VAE_CKPT` to that path. Elixir then uses `RecGPT.FSQ.load_params_from_vae_pt/1` so the same FSQ codebook as Python is used.

3. **Run eval and serve**  
   After building the fixture with the above:
   - **Serve:** `mix recgpt.serve` — recommendations use `RecGPT.Inference.forward/4` and `RecGPT.Decode.beam_search/4`. Set `RECGPT_FIXTURE` and checkpoint so the server loads the fixture built with .npy + VAE.
   - **Eval:** `mix recgpt.eval` loads state with `RecGPT.Serve.load_state/2` (fixture + checkpoint) and calls `RecGPT.Eval.evaluate/3` with test cases (e.g. from `test_sequences.json`). Use the same fixture (built with canonical texts and `--vae-ckpt`) so token_id_list and recommendations match.

So: **same embeddings (.npy) + same FSQ (from VAE)** = parity with the released model.

---

## Implementation details

- **FSQ from VAE:** `RecGPT.FSQ.load_params_from_vae_pt/1` loads the VAE .pt via `RecGPT.PtLoader.load!/1` and reads `quantizer.project_in.weight`, `quantizer.project_out.weight` (and biases). These are the same weights as in the released VAE. `RecGPT.FixtureBuild` uses them when you pass `opts[:vae_ckpt]` (e.g. `--vae-ckpt` or `RECGPT_VAE_CKPT`).
- **RecGPT export:** Exported from the RecGPT .pt only (GPT-2, wte, aux, pred_head). It does **not** include the VAE or FSQ. So for correct token_id_list you must pass the VAE .pt separately when building the fixture.
- **Embeddings:** See [26 Embedding mismatch](26_embedding_mismatch.md). For parity use the dataset’s `item_text_embeddings.npy`; Bumblebee is for new text or when not comparing to the released checkpoint.

---

## Summary

| Goal                        | Action                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Match released model**    | Build fixture with `--embeddings-npy` (dataset .npy) and `--vae-ckpt` (VAE .pt). Run eval/serve as usual.                                                     |
| **Why it was wrong before** | FSQ was taken from the RecGPT export (which has no FSQ) → dummy weights → wrong token IDs. And/or Bumblebee embeddings were used instead of the dataset .npy. |

---

## See also

- [26 Embedding mismatch](26_embedding_mismatch.md) — Our embeddings vs dataset .npy.
- [10 Parity by layer](10_parity_layers.md) — Per-layer parity and validation.
- [24 First step plan](24_first_step_plan.md) — First-step flow and VAE/ckpt options.
