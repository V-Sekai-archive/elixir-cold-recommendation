# Modality: Body

Part of [46_multimodal_foundation_encoders](../proposals/46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **body**. *(Archived.)*

---

## Summary

Body items (e.g. avatar, character, try-on, user shape) use [Anny](https://github.com/naver/anny)’s parameter vector (shape + pose) as the raw representation. A learned linear or MLP maps that vector to 768-d so it matches other modalities. That 768-d feeds the shared pipeline (common space, FSQ, aux). Segment ID marks positions as body.

---

## Encoder

- **Model:** [Anny](https://github.com/naver/anny) (NAVER differentiable human body model).
- **Input:** Anny’s **parameter vector** (shape + pose); no mesh encoder. Fixed-size vector per body.
- **Map:** Learned linear or MLP **param_dim → 768-d** to match other modalities.
- **Output:** 768-d. Same downstream path as vision/audio: body 768-d → (optional) body projector into common space → FSQ → 192-d → aux, or 768→192 → aux.

---

## Role in pipeline

- **Storage / indexing:** Store Anny params (or precomputed 768-d). For one codebook, body 768-d goes through the body projector into the common space.
- **Sequence:** For body positions, aux = 768-d from the body encoder (or 192-d after FSQ); token_id can be placeholder. **Segment ID:** `body`.
- **One codebook:** Add a body projector 768→768 into the common space; align using data where the same item has body + text/image/audio (e.g. avatar with description). One FSQ quantizes body-derived 768-d with text/image/audio.

---

## Dimension

| Output        | FuXi n_embd | Match |
|---------------|-------------|--------|
| 768 (param → proj) | 768         | Yes   |

---

## See also

- [46_multimodal_foundation_encoders](../proposals/46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- [Anny](https://github.com/naver/anny) — human body model (shape + pose params).
