# FuXi-Linear for Nanochat-Style Tasks — Investigation

**Goal:** Assess whether our FuXi-Linear (Elixir/Nx, in this repo) can be modified to handle the same tasks as [nanochat](https://github.com/karpathy/nanochat) (Python/PyTorch): pretraining, SFT, inference, and chat over a BPE-tokenized language model.

## Summary

**Yes.** FuXi-Linear is a decoder-only transformer with linear/complexity-efficient attention. With a bounded set of changes (configurable vocab, optional aux, LM training pipeline, and optional incremental state for decode), it can be used for nanochat-style next-token prediction and chat. FuXi does **not** use a growing KV cache: Retention uses a **fixed-size recurrent state** (S_t = decay·S_{t-1} + k^T v), so memory stays constant with sequence length. The main tradeoff is **attention mechanism**: nanochat uses standard causal (Flash) attention + RoPE; we use Retention + linear temporal/positional channels. Quality and scaling may differ; the architecture is compatible with LM tasks.

---

## 1. Nanochat (reference)

- **Repo:** [karpathy/nanochat](https://github.com/karpathy/nanochat) — “The best ChatGPT that $100 can buy.”
- **Stack:** Python, PyTorch, single-GPU or multi-GPU (e.g. 8× H100), minimal harness.
- **Stages:** Tokenization (BPE), pretraining, SFT, RL, evaluation (CORE, bpb), inference, chat (CLI + web UI).
- **Model:** Decoder-only GPT:
  - `wte`: token embedding (vocab_size × n_embd)
  - N × Block: **CausalSelfAttention** (Flash Attention 3 / SDPA, RoPE, QK norm, GQA, optional sliding window) + **MLP** (ReLU², 4× expansion)
  - `lm_head`: hidden → logits over vocab (untied)
- **Single dial:** `--depth` (number of layers); width, heads, LR, etc. derived for compute-optimal scaling.
- **Vocab:** BPE, default 32k; padded for efficiency.
- **Training:** Next-token prediction, cross-entropy; same for pretrain and SFT (teacher forcing).

---

## 2. Our FuXi-Linear (current)

- **Location:** `lib/recgpt/fuxi_linear_inference.ex` (and params in `fuxi_linear_inference_params.ex`).
- **Stack:** Elixir, Nx, EXLA (GPU).
- **Interface (RecGPT):**
  - **Inputs:** `batch_token_ids` (batch, seq_len), `batch_aux` (batch, seq_len, 192), `embed_mask`.
  - **Output:** logits (batch, 15_361) for last position; or (batch, seq_len, 15_361) for full sequence.
- **Architecture:**
  - Token embed: `wte` (15_361, 768) + **aux encoder** (192 → 768, required).
  - N × FuXi block: **Retention** (linear attention) + **LinearTemporalChannel** (time) + **LinearPositionalChannel** (position) + **MFFN** (SiLU, gated).
  - Final LN then **pred_head**: 768 → 15_361.
- **Vocab:** Fixed 15_361 (RecGPT FSQ semantic IDs).
- **Attention:** No softmax over full context; Retention + channel mixers (O(n) or chunked), sinusoidal position encoding.

So: same “shape” as a decoder-only LM (embed → blocks → head → logits), but different attention and a **required** aux path.

---

## 3. What Must Change for Nanochat-Style LM


| Concern                | Current (RecGPT)                                         | Change for LM                                                                                                                                       |
| ---------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vocab size**         | Fixed 15_361                                             | Configurable (e.g. 32_768, 50_257). `wte` and `pred_head` shapes keyed off `vocab_size`.                                                            |
| **Aux encoder**        | Required (ae.* params); 192-d per position               | **Optional.** When “LM mode”: pass zeros or skip; `forward_hidden` uses `token_embeds + 0` (or a conditional).                                      |
| **Token input**        | FSQ token IDs (from item sequences)                      | BPE token IDs from a tokenizer (no FSQ).                                                                                                            |
| **Training data**      | Item sequences → token_id_list, aux from item embeddings | Text → tokenize → (batch_token_ids, batch_aux=0, embed_mask=1). Reuse `forward_full_sequence` + CE loss.                                            |
| **Loss**               | `Training.loss_shifted_ce/2` over 15_361                 | Same; last dim = configurable vocab size.                                                                                                           |
| **Position encoding**  | FuXi sinusoidal in Retention + channel                   | Keep as-is for LM (no need for RoPE unless aiming for nanochat parity).                                                                             |
| **Scaling**            | Fixed n_blocks, n_embd                                   | Add config: `n_blocks`, `n_embd`, `vocab_size`, `max_seq_len` (and any head dims) so a single “depth” dial can drive a miniseries.                  |
| **Inference / decode** | Full-sequence forward; no KV-cache                       | For long generations, either: (a) full forward each step (simple, costly), or (b) implement KV-cache for Retention/linear layers for O(1) per step. |


---

## 4. What Stays the Same

- **Forward API:** `forward/5`, `forward_full_sequence/5` already return logits (batch, vocab) or (batch, seq_len, vocab); only vocab size becomes configurable.
- **Training:** Teacher-forced next-token CE; `build_train_batch` would be replaced (or bypassed) by a **text LM batch builder**: sample sequences from a corpus, tokenize, pack into (batch_token_ids, batch_aux=0, mask=1), call `forward_full_sequence`, then `loss_shifted_ce`.
- **Tasks:** Pretrain and SFT = same objective (next-token prediction). Eval (CORE, bpb) and chat UI are downstream of “logits over vocab”; they don’t care if the backbone is FuXi or GPT.

---

## 5. FuXi-Linear vs nanochat-style: how much better?

**Short answer:** There is no direct "FuXi vs nanochat" benchmark on **language modeling** (CORE, bpb, perplexity). So we can't say "X% better." The right way to see it is a **tradeoff**: nanochat-style is the default for **best LM quality**; FuXi-Linear is better for **efficiency and long context**, and may be slightly worse for raw LM quality at the same size.

| Dimension | Nanochat-style (causal + Flash + RoPE) | FuXi-Linear (Retention + channels) |
|-----------|----------------------------------------|------------------------------------|
| **LM quality** | Reference: established scaling (GPT-2, CORE leaderboard). Full softmax attention over context. | No published LM benchmarks. Linear attention often lags a bit on perplexity/CORE at same compute; Retention has done well in other domains (e.g. long sequence, recommendation). |
| **Compute (train)** | O(n²) in sequence length per layer. | O(n) per layer (Retention + linear channels). **Better** for long sequences. |
| **Memory** | O(n²) or O(n) with Flash; still grows with context. | O(n) state. **Better** for very long context. |
| **Inference speed** | One forward or KV-cache step; highly optimized (Flash, CUDA). | In our RecGPT setup, FuXi forward was **~33% faster** than GPT-2 forward (see `docs/archived/89_fuxi_latency_log.md`). That's same task (recommendation), not LM. |
| **Decode** | KV-cache is standard; O(1) per new token. | No KV-cache in our FuXi yet; full forward each step unless we add a Retention cache. So **worse** for long autoregressive decode until we implement it. |

So: FuXi-Linear is **better** on speed and memory for long sequences and fits "same LM task, different architecture." It is **not** proven "better" than nanochat-style on LM quality; for "best CORE / best bpb" you'd still bet on nanochat's stack. For "train and chat with linear attention and lower cost/longer context," FuXi is a good candidate.

---

## 5.5. Context scaling: Gemini million+ vs FuXi-Linear upstream

| Dimension | Gemini (1.5 Pro / 2.5 Pro) | FuXi-Linear upstream (paper) | Our FuXi-Linear (elixir-recgpt) |
|-----------|----------------------------|------------------------------|----------------------------------|
| **Context window** | **1M–2M tokens** (1.5 Pro 2M; 2.5 Pro 1M standard, 2M in testing) | **Several thousand** (paper: "thousand-length scale", "sequences of several thousand tokens"; industrial RS "exceeding 10⁴ interactions") | **No hard cap** in code; seq_len from input; Channel P uses sinusoidal (any length) when no learned emb, else slice to `max_seq_len` from init; `chunk_size` for prefill memory |
| **Attention** | Causal Transformer; softmax attention | Linear: Retention + Temporal + Positional channels | Same as upstream (Retention + Channel T + Channel P) |
| **Prefill cost** | O(L²) in context length | O(L) (linear attention) — paper: **up to 10× prefill speedup** vs baselines | O(L) full forward; O(L) per chunk when `chunk_size` set |
| **Decode (per token)** | O(L): KV cache read/write, one new K,V | O(1) recurrent (paper: **up to 21× decode speedup**) | O(1) Retention + Channel T; O(L) Channel P (cached v) |
| **Memory vs context** | **O(L)** — KV cache grows with context | O(1) for Retention + Temporal state; O(L) only if positional channel caches history | O(1) Retention + Channel T state; O(L) Channel P cache (per block) |
| **Use case** | General long-context LM (documents, code, chat) | Long **sequential recommendation** (user histories, time-aware); power-law scaling at 1k–10k length | RecGPT recommendation + optional LM; same scaling as upstream |

**Summary**

- **Gemini** targets **million-token** context with a growing KV cache: prefill is quadratic, decode is linear in context length, and memory grows with context. That suits maximum context and exact attention over the full window.
- **FuXi-Linear upstream** is designed for **thousand-scale** (and up to ~10k) sequences with **linear** prefill and **constant-time** decode (recurrent state). It is validated on sequential recommendation, with 10× prefill and 21× decode speedups vs quadratic baselines.
- **Our port**: nothing in the architecture prevents million-token (or longer) sequences. We scale linearly (prefill O(L), decode O(1) for Retention/Channel T, O(L) for Channel P cache). The paper only validated at thousands; we have not run at 1M. Practical limits are **memory** (Channel P caches v per position, O(L)) and, if using learned Channel P positional emb, **init** with `max_seq_len` ≥ desired length—or omit it and use sinusoidal, which extends to any length. So our port is **compatible with very long context** (e.g. million-token) given enough RAM and, if needed, sinusoidal Channel P or a large `max_seq_len` at init.

---

## 6. Nanochat features we don't need to mirror (optional)


- **RoPE:** FuXi already has position; can keep sinusoidal.
- **Flash Attention / exact attention:** We keep linear attention; different quality/speed tradeoff.
- **GQA / sliding window:** Not in FuXi; could be added later if needed.
- **Value embeddings (ResFormer), resid_lambdas, x0_lambdas:** Nanochat-specific; not required for “FuXi does LM.”

---

## 7. Recommended implementation steps

1. **Configurable model size**
  - Add (e.g.) `vocab_size`, `n_blocks`, `n_embd`, `max_seq_len` to FuXi init/params.
  - Build `wte` and `pred_head` from `vocab_size`; keep `n_embd` (768 or match nanochat width).
2. **Optional aux**
  - In `forward_hidden`, if `batch_aux` is `nil` or a flag says “LM mode”, set `aux_768 = 0` (or skip `apply_aux_encoder`); else keep current behavior. This keeps RecGPT unchanged.
3. **LM training pipeline**
  - New (or reused) data pipeline: text corpus → BPE tokenizer → sequences of token IDs.
  - Build batches: (batch_token_ids, batch_aux = zeros or nil, embed_mask = ones).
  - Use existing `forward_full_sequence` + `loss_shifted_ce` with the new vocab size.
4. **Tokenizer**
  - Integrate or export BPE (e.g. train with a small script, or use a pre-trained tokenizer and export vocab + merges). Nanochat uses a GPT-2–style BPE; we only need token IDs for training/inference.
5. **Inference for chat**
  - Short sequences: full forward each step is acceptable.
  - Longer generation: design a KV-cache for FuXi’s Retention (and any other stateful linear attention) so we decode one token at a time without re-running the full sequence.
6. **Eval**
  - CORE / bpb: same as nanochat (run on eval set, compute metric). No change to FuXi itself.

---

## 7.5. Strategy for multimodal (text + vision / audio)

**Goal:** Use FuXi-Linear for inputs that mix text with other modalities (e.g. images, audio) in one sequence, without changing the core block layout. For recommended foundational encoders per modality and dimension alignment (768-d), see [46_multimodal_foundation_encoders](46_multimodal_foundation_encoders.md).

**Options (in order of fit with current code):**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Aux path as modality embedding** | Keep `batch_aux` (batch, seq_len, 192). For text-only positions: zeros or learned “text” vector. For image (or other) positions: project encoder output (e.g. CLIP/ViT patch or global embedding) to 192-d and fill `batch_aux`. FuXi already does `token_embeds + apply_aux_encoder(aux)` so image info is additive to token embeddings. | No change to FuXi blocks; reuse RecGPT aux encoder. Linear cost in seq_len. | Aux is 192-d and additive only; no explicit cross-attention between modalities. Best for “one vector per position” (e.g. one image → one or a few aux rows). |
| **B. Interleaved tokens (projected)** | Vision encoder (e.g. ViT) outputs patch tokens; project to `n_embd` and treat as a second “vocab” or special token IDs. Build one sequence: e.g. `[img_1, …, img_k, text_1, …, text_m]`. Use `batch_aux` for text positions (or zeros) and a different projection for image positions so the model sees both. Alternatively: single embedding table with text vocab + “image patch” entries; positions and `all_timestamps` encode order. | Single sequence, one forward; Channel T/P see full order. | Need a clear convention for segment (image vs text) and position; may need extra projectors and token-type logic. |
| **C. Separate encoders + concat sequence** | Encode image to a fixed number of vectors (e.g. 256 patches → 256 × n_embd); encode text to token embeddings. Concat: `hidden = [image_embed; text_embed]`, then run FuXi on the full sequence. Use `all_timestamps` to mark image vs text (e.g. image positions 0..k-1, text k..L-1) so Channel T can treat them differently if desired. | Clear separation of modalities; flexible length per modality. | Need to define “timestamps” for non-time modalities (e.g. position indices or segment IDs); aux path could still carry per-position modality flags. |

**Recommended path (minimal change):**

1. **Phase 1 — Aux as image (or other) side info**  
   - Add a **projector** (e.g. linear or MLP) from vision encoder output (e.g. 768-d or patch sequence) to 192-d.  
   - For each position that “sees” an image, set `batch_aux[b, pos, :]` to that projected vector; for text-only positions, keep aux zeros (or a learned text bias).  
   - No change to FuXi blocks; only data pipeline and projector. Good for: image captioning, single-image QA, or “one embedding per image” in a sequence.

2. **Phase 2 — Multiple “image tokens” per image**  
   - Vision encoder outputs many patch tokens (e.g. 256). Project each to 192-d and assign to a contiguous span of positions.  
   - Sequence layout: e.g. `[img_pos_1, …, img_pos_K, text_pos_1, …, text_pos_M]`. Use a single `batch_token_ids` that either (a) uses special IDs for image positions and a small “image embedding” table, or (b) uses a single shared embedding and different `batch_aux` per position (image vs text).  
   - Set `all_timestamps` (e.g. position index or segment id) so Channel T gets a consistent notion of order; Channel P already gives position-in-sequence.

3. **Phase 3 — Audio or other modalities**  
   - Same idea: encoder (e.g. HuBERT, wav2vec) → project to 192-d (or to n_embd and feed as “tokens”).  
   - Either feed as aux for existing positions, or add a second token stream and interleave with text (same as B/C).  
   - Timestamps for audio can be real (seconds) or frame indices; FuXi’s Channel T supports real timestamps.

**Summary:** Multimodal fits the current FuXi interface by (1) using the existing **aux path** for modality embeddings (recommended first step), and (2) optionally moving to **interleaved token sequences** with a shared embedding space and clear position/timestamp conventions. No change to Retention / Channel T / Channel P math; only input construction and optional projectors.

**Is the aux path big enough?** The aux input is fixed at **192 dimensions** (batch, seq_len, 192). That comes from RecGPT’s FSQ design (4 × 192-d codes per item), not from vision. Typical encoders output larger vectors:

| Encoder | Typical output dim | Fit into 192-d aux |
|--------|---------------------|---------------------|
| CLIP (ViT-B) | 512 | Project 512→192 (compression) |
| CLIP (ViT-L), ViT-B/16 | 768 | Project 768→192 (strong compression) |
| ViT-L, LLaVA patch tokens | 1024 | Project 1024→192 (very strong compression) |

So **192 is a bottleneck** for rich vision: a single linear 768→192 (or 512→192) loses a lot of capacity. It can still work for “light” conditioning (e.g. one global image vector for captioning), but for many patches or high-fidelity vision you have two better options:

1. **Make aux dim configurable**  
   Add `aux_dim` (e.g. 192, 512, or 768) to init; `ae.linear` becomes `(aux_dim, n_embd)`. Vision (or other modality) encoders then output `aux_dim`-d directly, or you project to `aux_dim` with less compression. Keeps the “one vector per position” design.

2. **Bypass aux for vision**  
   Use **interleaved tokens** (approach B/C): project image patches to `n_embd` (768) and feed them as token embeddings (with or without a small amount of aux). No 192-d cap; capacity is the same as the model’s hidden size.

**How bad is bypassing aux (interleaved tokens)?** Quality-wise it’s usually **better** than squeezing vision into 192-d aux; you keep full resolution per patch and match the LLaVA/Flamingo pattern. The cost is elsewhere: (1) **Longer sequences** — e.g. 256 patch tokens per image plus text, so more FLOPs and O(L) memory for Channel P cache. (2) **New input pipeline** — you need a convention for “image token” vs “text token” (separate embed table, reserved ID range, or segment flags) and to build the interleaved sequence and `all_timestamps` correctly. (3) **Training** — the image→768 projector (and optionally the rest of the model) must be trained or fine-tuned on multimodal data; you can’t just plug in a frozen CLIP. So “bypass aux” is not bad for quality or for the FuXi block (it only sees 768-d hidden states); it’s a bit more work and more compute per example than “one global image in aux.”

Recommendation: for **single global image** (e.g. image captioning), 192-d aux plus a learned 768→192 (or 512→192) projector is acceptable. For **many image tokens or high fidelity**, make `aux_dim` configurable (e.g. 768) or use interleaved tokens so the model sees full-dimensional modality embeddings.

---

## 8. Conclusion

FuXi-Linear can be modified to handle nanochat’s **tasks** (pretrain, SFT, eval, inference, chat) by:

- Making vocab and model size configurable,
- Making the aux path optional for pure text LM,
- Adding a text → BPE → batch pipeline and reusing existing CE training,
- Optionally adding a KV-cache for efficient autoregressive decode.

**Multimodal** (text + image/audio) can be added without changing the FuXi blocks: use the existing **aux path** for modality embeddings (e.g. project vision encoder output to 192-d), or **interleave** projected modality tokens with text in one sequence and set `all_timestamps` for order (see §7.5).

The **architecture** remains FuXi (linear attention + channel mixers); we do not need to reimplement nanochat’s GPT block-for-block. That implies different scaling and possibly different quality than nanochat’s Flash-based GPT at the same “depth,” but the same **task** (next-token LM and chat) is supported.

**References**

- [nanochat](https://github.com/karpathy/nanochat) — Karpathy’s minimal LLM harness.
- FuXi-Linear (this repo): `lib/recgpt/fuxi_linear_inference.ex`, `lib/recgpt/training.ex`.

