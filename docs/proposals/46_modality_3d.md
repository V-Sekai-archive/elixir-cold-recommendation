# Modality: 3D (TRELLIS.2)

Part of [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md). One encoder per modality; this doc is **3D**.

---

## Summary

For each 3D catalogue item we need one 768-d embedding aligned with MPNet for retrieval and for the shared pipeline (FSQ, aux). We reuse the MPNet-style dual-encoder recipe: one side is text (MPNet, frozen), the other is the TRELLIS.2 structured latent passed through a learned map to 768-d. Contrastive loss trains the map so that latent and text for the same object sit close in 768-d space.

---

## Storage: keep the latent as-is

- **Source:** [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (mesh → O-Voxel → SC-VAE encode).
- **Stored form:** The **structured latent as-is** (e.g. ~1.2K tokens × channels, on the order of tens to ~150 KB per object). Do not resize or re-encode for storage; this is the canonical 3D representation.

---

## Encoding: pool + project to 768-d

We use **one** 768-d vector per item. Do not patchify or feed many 3D tokens into FuXi for this embedding; FSQ is per item, so one vector per item is enough.

1. **Pool:** Treat the latent as (T, C). **Mean-pool over T** → shape (C,).
2. **Project:** Learned **Linear(C, 768)** or small **MLP(C → 768)**. Optional L2-normalize to match MPNet.
3. **Output:** This 768-d vector is the 3D item embedding used for retrieval, contrastive alignment with text, FSQ, and aux.

The many-token path (per-token project → FuXi → collapse to one vector) is not used here; it is heavier and better suited to settings where one model does fine-grained reasoning over multi-item or long 3D sequences.

---

## Training: contrastive (latent vs text)

- **Pairs:** (structured latent for object O, text for object O). The text is the **text input to TRELLIS.2** (e.g. text-to-3D prompt) or a caption for the 3D asset.
- **Loss:** Contrastive between **map(latent)** (768-d) and **MPNet(text)** (768-d). Same item → pull close; different items → push apart. Only the map is trained; MPNet is frozen.
- **Result:** The map’s 768-d output lives in the same space as MPNet for retrieval and fusion.

---

## Role in pipeline

- **Stored:** Exact TRELLIS.2 latent.
- **Used:** map(latent) = 768-d → common space (if using one codebook) → FSQ → 192-d → aux, or 768→192 → aux, or aux_dim = 768.
- **Segment ID:** `3D` so the model knows the position is 3D.

---

## Dimension

| Output | FuXi n_embd | Match |
|--------|-------------|--------|
| 768 (latent → map) | 768 | Yes |

---

## See also

- [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md) — overview, aux, one codebook, segment IDs.
- [46_modality_text](46_modality_text.md) — text encoder (MPNet).
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — 3D structured latent (O-Voxel, SC-VAE).
