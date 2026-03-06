# Modality: Vision

Part of [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **vision**.

---

## Summary

Image items use DINOv2 ViT-B to get one 768-d vector per item (patch or [CLS] features). Output matches FuXi `n_embd`; for ViT-L/ViT-g add a linear projector to 768-d. That 768-d (or 768→192 after FSQ) feeds the shared pipeline (aux, sequence). Segment ID marks positions as image. Train with contrastive loss on paired image–text data. **Preferred datasets** (each includes image + caption in one load): `mrzjy/AniGamePersonaCaps` (anime/character), `nlphuji/flickr30k` (general); see Training data below.

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

## Training data (image–text pairs for contrastive)

Train the vision encoder (or vision→768-d projector) with **contrastive loss**: image → DINOv2 → 768-d, text → MPNet → 768-d; same-item pairs pulled close, different-item pairs pushed apart.

**Preferred datasets (both include image + caption in the same dataset):**

| Role | Dataset | Vision | Caption |
|------|---------|--------|---------|
| Anime / character | **mrzjy/AniGamePersonaCaps** | [HF](https://huggingface.co/datasets/mrzjy/AniGamePersonaCaps) — `image` (PIL) | `title`, `description`, `caption.appearance`, `caption.personality` |
| General | **nlphuji/flickr30k** | [HF](https://huggingface.co/datasets/nlphuji/flickr30k) — `image` (PIL) | multiple captions per image |

Load with:

```python
load_dataset("mrzjy/AniGamePersonaCaps")
load_dataset("nlphuji/flickr30k")
```

Use one caption per image (e.g. `caption.appearance.human` or first caption in list). No URL fetch or join required.

---

## Dimension

| Output   | FuXi n_embd | Match |
|----------|-------------|--------|
| 768 (ViT-B) | 768         | Yes   |

---

## See also

- [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) §7.5 — multimodal strategy.
