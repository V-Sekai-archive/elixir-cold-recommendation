# Track: Multimodal / avatar-asset (B)

**Product-first:** A rope bridge — narrow, distinctive, one vertical at a time. First movers (avatar and asset marketplaces) cross; later you reinforce or add lanes.

This doc extracts **Track B** from [CHIBIFIRE_TRACKS_IMPROVEMENT](../archived/CHIBIFIRE_TRACKS_IMPROVEMENT.md) (chibifire.com tracks) for execution.

---

## Product (user-facing)

- **"Recommend by image"** and avatar/asset discovery for VRChat- and Booth-style catalogs (vision and 3D).
- One vertical at a time: avatar marketplaces, asset stores, and character/catalog discovery where items are images, 3D models, or optionally text.

---

## Why rope bridge

- **Narrow:** A single vertical (avatars, UGC assets) before expanding.
- **Pioneering:** Few recommendation systems focus on avatar/3D and image in this niche.
- **First-mover:** You define the crossing (recommend by image, similar avatars/assets); then reinforce or add lanes.

---

## Foundation: text embedding pipeline (already in place)

The **text** embedding pipeline already exists: MPNet (`sentence-transformers/all-mpnet-base-v2`) via `RecGPT.Embedding`, 768-d → FSQ → token sequence, used for catalog items and sequences. This track **adds** vision and 3D to the same pipeline so items can be text, image, or 3D in one recommendation system.

---

## Scope (text + vision + 3D)

| Layer | What |
|-------|------|
| **Text** | Existing: MPNet → 768-d, FSQ → tokens; `RecGPT.Embedding`, fixture build, pretrain. |
| **Vision** | DINOv2 ViT-B → 768-d per image; vision projector (768→768) trained with contrastive loss against MPNet text. Enables "recommend by image" and image–text alignment. |
| **3D** | TRELLIS.2-style 3D latent → one 768-d vector (pool + project); contrastive against text. Same common space as vision and text for one codebook. |
| **Pipeline** | Shared FuXi sequence: text (MPNet), image (DINOv2 + projector), 3D (pool + project); segment IDs per modality; aux 768-d or 192-d FSQ. |

---

## Technical references (this repo)

- **Text (existing):** `RecGPT.Embedding` — MPNet, 768-d, L2 norm; `encode_item_text_dict/1`; used in fixture build and pretrain. See [46_modality_text](../archived/46_modality_text.md).
- **Vision:** [46_modality_vision](46_modality_vision.md) — DINOv2, preferred datasets (AniGamePersonaCaps, flickr30k), contrastive training.
- **3D:** [46_modality_3d](46_modality_3d.md) — TRELLIS.2 latent, pool + project to 768-d, contrastive against text.
- **Overview:** [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md) — one encoder per modality, single pipeline, segment IDs.
- **Implementation:** `RecGPT.VisionProjector`, `RecGPT.VisionContrastive`, `mix recgpt.train_vision_contrastive`, `scripts/download_vision_contrastive_data.py`.

---

## Target audience

- Avatar and character marketplaces (e.g. VRChat, Booth.pm-style).
- UGC asset stores where items are images or 3D models.
- Catalogs that want "similar to this image" or "similar to this asset" in a unified recommendation system.

---

## Improvement score (from parent doc)

| Criterion | Score |
|-----------|:-----:|
| Capability lift | 5 |
| Brand lift | 5 |
| Sustainability lift | 4 |
| Ecosystem lift | 4 |
| Leverage for next tracks | 4 |
| Reach | 3 |
| **Improvement score** | **4.3** (rank 3 in full table) |

---

## See also

- [CHIBIFIRE_TRACKS_IMPROVEMENT](../archived/CHIBIFIRE_TRACKS_IMPROVEMENT.md) — full track list, scores, and ranking.
- [46_modality_vision](46_modality_vision.md), [46_modality_3d](46_modality_3d.md), [46_modality_text](../archived/46_modality_text.md) — modality specs and training data.
