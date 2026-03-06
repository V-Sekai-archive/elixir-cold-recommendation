# Modality: Audio

Part of [46_multimodal_foundation_encoders](../proposals/46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **audio**. *(Archived.)*

---

## Summary

Audio items use WavLM base to get one 768-d vector per item (frame-level or pooled). Output matches FuXi `n_embd`. That 768-d (or 768→192 after FSQ) feeds the shared pipeline (aux, sequence). Segment ID marks positions as audio.

---

## Encoder

- **Model:** WavLM base (e.g. `microsoft/wavlm-base`).
- **Input:** Raw audio, 16 kHz waveform; frame-level or pooled over time.
- **Output:** 768-d per frame or pooled. Matches FuXi `n_embd`.
- **Use:** 768-d as the item embedding; feed into aux (768-d or 192-d after FSQ) or into the common space when using one codebook.

---

## Role in pipeline

- **Storage / indexing:** Encode with WavLM → 768-d (or 768→192 if encoding like FSQ). Store as item embedding.
- **Sequence:** For audio positions, aux = 768-d from WavLM (or 192-d projected); token_id can be placeholder. **Segment ID:** `audio`.
- **One codebook:** Add an audio projector 768→768 into the common space; align with contrastive loss (same item in audio + text close).

---

## Dimension

| Output   | FuXi n_embd | Match |
|----------|-------------|--------|
| 768 (base) | 768         | Yes   |

---

## See also

- [46_multimodal_foundation_encoders](../proposals/46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) §7.5 — multimodal strategy.
