# Multimodal foundation encoders and single-pipeline design

Sub-proposal of the [documentation index](../README.md). One encoder per modality (text, vision, audio, body, 3D), one 768-d sequence into FuXi, optional segment IDs. Complements [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) §7.5.

---

## Per-modality docs (divide and conquer)

Active proposal docs (in `docs/proposals/`): **Vision** and **3D** (encoders, training data, pipeline). Text, audio, and body are summarized below and detailed in archived docs.

| Modality | Doc |
|----------|-----|
| **Text** | [46_modality_text](../archived/46_modality_text.md) — MPNet, 768-d, already in use. *(archived)* |
| **Vision** | [46_modality_vision](46_modality_vision.md) — DINOv2 ViT-B, 768-d; preferred datasets (image + caption): AniGamePersonaCaps, flickr30k. |
| **Audio** | [46_modality_audio](../archived/46_modality_audio.md) — WavLM base, 768-d. *(archived)* |
| **Body** | [46_modality_body](../archived/46_modality_body.md) — Anny params → 768-d. *(archived)* |
| **3D** | [46_modality_3d](46_modality_3d.md) — TRELLIS.2 latent as-is, one map → 768-d (the output we want), contrastive vs text. |

---

## Problem or limitation

We want multimodal zero-shot (text + image + audio + body + 3D) without maintaining separate pipelines or changing FuXi's core. Today:

- **Text** is covered (MPNet via `RecGPT.Embedding`, 768-d).
- **Vision and audio** have no designated encoders; §7.5 describes options (aux vs interleaved) but not concrete model choices or dimension alignment.
- **Body and 3D** are not yet specified.
- **Two pipelines** (aux-only vs interleaved) would duplicate logic; we need one sequence builder that can fill each position from any modality.
- **Dimensions** must match FuXi `n_embd` (768) so no ad-hoc projectors are required for base variants.

---

## Proposed improvement

1. **One foundational encoder per modality** (all 768-d for base variants): see [per-modality docs](#per-modality-docs-divide-and-conquer) above.
2. **Single pipeline:** One sequence of positions. Per position: token_id (or placeholder) and **aux = item embedding** (FSQ 192-d, or 768-d when aux_dim = 768). So `hidden[pos] = WTE[token_id] + apply_aux_encoder(aux)` for all items; aux carries the modality-specific item representation.
3. **Optional segment IDs:** Learned embedding per type (text / image / audio / body / 3D) added to the 768-d at each position; ~1–3% gain, clarifies token type.
4. **Aux:** Keep for item side info; **upgrade** or encode to 192-d (see below).

---

## Multimodals are items: upgrade aux

In recommendation and catalog settings, **multimodal content is also items**. Treat vision, audio, body, and 3D as **item representations** in **aux**:

- **Make aux_dim configurable** (192 for backward-compat / FSQ-only, or 768 for unified multimodal items). Init: `ae.linear` has shape `(aux_dim, n_embd)`.
- **Unified item embedding in aux:** For each position, aux holds the *item* embedding: 192-d FSQ for text, or 768-d from the modality encoder (DINOv2, WavLM, Anny proj, TRELLIS.2 map, etc.). **Segment IDs** indicate item modality (text / image / audio / body / 3D).

So: **upgrade aux** to 768 (no compression), or **encode** each modality into 192-d like FSQ (keep aux at 192-d). See "Encode into 192-d like FSQ" and "One codebook" below.

---

## Encode into 192-d like FSQ (no aux upgrade)

FSQ already **encodes** the text embedder (MPNet 768-d) into 192-d. We can do the same for other modalities: **encode** encoder output into 192-d so aux stays 192-d.

- **Text:** MPNet 768-d → FSQ → 192-d. Already in place.
- **Vision / Audio / Body / 3D:** 768-d → learned 768→192 (or one codebook) → 192-d → aux.

**Tradeoff:** Encoding 768→192 loses information. Either **upgrade aux to 768**, or **encode into 192-d like FSQ**.

---

## One codebook for all modalities

1. **Common 768-d space.** Modality projectors (768→768) map each modality’s 768-d into one shared space.
2. **Align the projectors.** Contrastive (or joint) loss: same item in different modalities → close; different items → far.
3. **One FSQ on the common 768-d.** One codebook quantizes common 768-d → 192-d → aux. Segment ID still per position.
4. **Body and 3D:** Add body projector and 3D map output to common space; same FSQ. See [46_modality_body](../archived/46_modality_body.md) and [46_modality_3d](46_modality_3d.md).

**Summary:** Modality encoders → 768-d → modality projectors → common 768-d → one FSQ → 192-d → aux.

---

## How we know modality type (segment IDs)

With all modalities as 768-d vectors, the model cannot tell the type from the vector alone. **Segment IDs:** assign a modality type per position and add a learned segment embedding to the 768-d. Sequence builder sets segment (e.g. "text", "image", "audio", "body", "3D"). So we know what type each 768-d is by construction.

---

## Is it worth dropping aux?

**No.** Keep aux and **upgrade** it (or encode to 192-d).

- **Aux** is the unified *item embedding* channel. **Segment IDs** indicate item modality (text / image / audio / body / 3D). Make aux_dim configurable (768) so multimodals can be items without compression.
- So: **segment IDs = item modality type; aux = item embedding.** Do not drop aux.

---

## Dimension alignment (reference)

| Modality | Encoder            | Output | FuXi n_embd | Match |
|----------|--------------------|--------|-------------|--------|
| Text     | MPNet              | 768    | 768         | Yes    |
| Vision   | DINOv2 ViT-B       | 768    | 768         | Yes    |
| Audio    | WavLM base         | 768    | 768         | Yes    |
| Body     | Anny params → proj | 768    | 768         | Yes    |
| 3D       | TRELLIS.2 latent → map | 768    | 768         | Yes    |

Aux: `aux_dim` input (192 or 768), projected to 768 inside FuXi (`apply_aux_encoder`). Upgrade: make aux_dim configurable.

---

## Recommended improvements

- [ ] **Per modality:** [Vision](46_modality_vision.md), [3D](46_modality_3d.md) (active); [Text](../archived/46_modality_text.md), [Audio](../archived/46_modality_audio.md), [Body](../archived/46_modality_body.md) (archived).
- [ ] **Aux path (choose one):** (a) Upgrade aux to 768; or (b) Encode like FSQ (768→192 per modality); or (c) One codebook (projectors → common space → one FSQ → 192-d).
- [ ] **Sequence builder:** Single function: per position, token_id + aux = item embedding (192-d or 768-d); produces `(batch, seq_len, 768)` hidden and `(batch, seq_len, aux_dim)` aux.
- [ ] **Segment IDs (optional):** Learned modality embeddings; add to hidden per position. ~1–3% gain.
- [ ] **Docs:** Point §7.5 to this proposal; keep §7.5 for strategy (aux vs interleaved, phases).

---

## See also

- [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) §7.5 — Multimodal strategy (aux vs interleaved, aux dim).
- [22 Top-tier recommendations](22_top_tier_recommendations.md) — Problem/improvement format.
- `lib/recgpt/fuxi_linear_inference.ex` — `n_embd` 768, `batch_aux` 192.
- `lib/recgpt/embedding.ex` — MPNet text embedder.
