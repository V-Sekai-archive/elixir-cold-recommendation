# Modality: Vision

Part of [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **vision**.

---

## Summary

Image items use DINOv2 ViT-B to get one 768-d vector per item (patch or [CLS] features). Output matches FuXi `n_embd`; for ViT-L/ViT-g add a linear projector to 768-d. That 768-d (or 768→192 after FSQ) feeds the shared pipeline (aux, sequence). Segment ID marks positions as image.

---

## Encoder

- **Model:** DINOv2 ViT-B (e.g. `facebook/dinov2-base`).
- **Input:** Image (e.g. product photo).
- **Output:** 768-d per patch or [CLS]. Matches FuXi `n_embd` for ViT-B; for ViT-L or ViT-g, add a linear projector to 768-d.
- **Use:** Patch or [CLS] as the item embedding; feed into aux (768-d or 192-d after FSQ) or into the common space when using one codebook.

---

## Role in pipeline

- **Storage / indexing:** Encode with DINOv2 → 768-d (or 768→192 if encoding like FSQ). Store as item embedding.
- **Sequence:** For image positions, aux = 768-d from DINOv2 (or 192-d projected); token_id can be placeholder. **Segment ID:** `image`.
- **One codebook:** Add a vision projector 768→768 into the common space; align with contrastive loss (same item in image + text close).

---

## Dimension

| Output   | FuXi n_embd | Match |
|----------|-------------|--------|
| 768 (ViT-B) | 768         | Yes   |

---

## See also

- [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) §7.5 — multimodal strategy.
