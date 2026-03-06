# Modality: Text

Part of [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **text**.

---

## Summary

Text items use MPNet to get one 768-d vector per item. No extra projector is needed; output already matches FuXi `n_embd`. That 768-d (or FSQ 768→192) feeds the shared pipeline (aux, sequence). Segment ID marks positions as text.

---

## Encoder

- **Model:** `sentence-transformers/all-mpnet-base-v2` (MPNet).
- **Implementation:** `RecGPT.Embedding` (Bumblebee): mean pooling over token embeddings, L2-normalize. RecGPT uses `recgpt_item_text/1` for dataset parity.
- **Input:** Item text (e.g. title, description).
- **Output:** 768-d vector. Matches FuXi `n_embd`; no projector needed.

---

## Role in pipeline

- **Storage / indexing:** Use MPNet 768-d directly, or FSQ 768→192 (or token IDs) for aux.
- **Sequence:** For text positions, `hidden[pos] = WTE[token_id] + apply_aux_encoder(aux)`; aux is 192-d FSQ or 768-d when aux is upgraded.
- **Segment ID:** `text`.

---

## Dimension

| Output | FuXi n_embd | Match |
|--------|-------------|--------|
| 768    | 768         | Yes   |

---

## See also

- [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- `lib/recgpt/embedding.ex` — MPNet text embedder.
