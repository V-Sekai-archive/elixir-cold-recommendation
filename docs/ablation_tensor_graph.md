# Ablation testing: tensor graph and decode path

Goal: identify components that can be removed or simplified to improve latency **without breaking**:
1. **Semantic id** — the 4-token path through the trie that maps to a single item_id.
2. **Item fetch top-k** — returning up to `top_k` distinct item_ids, ordered by score.
3. **Pretraining** — we must still be able to pretrain the model (or the ablation is inference-only and leaves the trained checkpoint unchanged).

Reference: [latency_flow.md](latency_flow.md) for the full E2E and GPU tensor graph.

**Pretraining compatibility:** Inference-only ablations (e.g. skip aux path when aux=0, skip prefix_tokens transfer, beam_width override) do not change the model or checkpoint; pretraining continues to use the full graph (e.g. aux encoder in training). Ablations that remove a component from the *model* (e.g. delete aux encoder entirely) would require pretraining again without that component and a new checkpoint.

**Methodology:** Remove or change one component at a time and re-measure so the impact is not confounded. Report both latency (e.g. mean and P50/P99 from `mix recgpt.trace_predict --runs N`) and quality (Hit@k, MRR from `mix recgpt.eval`) when comparing baseline vs ablated config. Re-test with different context lengths and top_k where relevant (e.g. beam ablation affects steps 1–3 more for larger beam).

---

## Tensor graph (what runs on GPU/CPU)

| Stage | What it does | Required for semantic id? | Required for top-k? | Ablation idea |
|-------|------------------------|---------------------------|---------------------|---------------|
| **Context → tokens** | `item_id_to_tokens` gather, shape `{1, context_len}` | Yes — context drives next-token distribution | Yes | None; must feed model correct context. |
| **Embed** | `wte[token_ids]` → token embeddings | Yes | Yes | No; core input. |
| **Aux encoder** | `apply_aux_encoder(aux_192, mask, params)` → add to embeddings; we pass aux=0, mask=1 | **Maybe** — with aux=0, mask=1 this is a learned constant (bias after LN). Removing → different logits. | Same | **Ablate:** run with `aux_768 = 0` (skip add). Compare item_ids vs baseline; if identical or quality OK, can remove aux path in Defn for speed. |
| **WPE** | Add position embeddings | Yes — transformer needs positions | Yes | No. |
| **12× block** | Attention + MLP, with cache (full) or incremental (KV append) | Yes — core model | Yes | No; would need distilled smaller model to reduce layers. |
| **LN + head** | Final layer norm and logits projection | Yes | Yes | No. |
| **Trie restriction** | `next_state` / `item_at_leaf` gather, valid_mask, select(valid, logits, -∞) | Yes — only valid next tokens → valid 4-token paths → valid item_ids | Yes | No; without trie we get invalid token sequences. |
| **Beam search** | Steps 0–3, beam_width candidates, top_k per step | Step 0→3 required to get 4 tokens. Beam width ≥ 1 required for ≥1 item. | top_k > 1 needs beam ≥ 2 to get multiple candidates | **Ablate:** beam_width=1 (greedy) vs current: faster (smaller batch in steps 1–3) but only one candidate; then need multiple requests or different strategy for top_k. |
| **Sync** | `to_flat_list(item_ids)`, `to_flat_list(beam_scores)`, `backend_transfer(prefix_tokens)` | item_ids and scores required. prefix_tokens only for fallback when item_id == -1. | Same | **Ablate:** skip transferring prefix_tokens to host when all item_ids ≥ 0 (no trie fallback). Saves one transfer; measure. |
| **Post-decode** | Zip, trie fallback (map lookup by 4 tokens), sort, uniq, take(top_k) | item_id resolution (trie or map) required for semantic id | Yes | Fallback only when item_at_leaf returned -1; if rare, already cheap. |

---

## What must stay (cannot remove without breaking)

- **Pretraining:** Training code (e.g. `RecGPT.Training`, `RecGPT.AxonTrain`, inference used in loss) must keep the full graph so we can pretrain; inference-only shortcuts (e.g. skip_aux_encoder when aux=0) do not touch training.
- **Context → tokens** and **embed**: model input.
- **WPE**: position information.
- **All 12 blocks + LN + head**: model body; changing depth requires a different model.
- **KV cache**: required for correct incremental (steps 1–3); cannot drop.
- **Trie (next_state, item_at_leaf)**: ensures every 4-token path is valid and maps to an item_id.
- **Four steps (0–3)**: one item = 4 tokens; fewer steps would change the semantic id format.
- **item_ids and scores to host**: needed to form the response.
- **Resolve item_id** (from item_at_leaf or trie map fallback): needed for semantic id and top-k list.

---

## Ablation candidates (safe to try)

### 1. Aux encoder path (inference)

- **Current:** We pass `aux = 0`, `mask = 1`. Defn does `aux_768 = linear(aux) + bias; LN; * mask` → adds a learned constant to embeddings.
- **Ablate:** In Defn or in serve, pass `combined = token_embeds` (skip adding `aux_768`) when aux is zero, or add a flag to skip `apply_aux_encoder` when aux is all zeros.
- **Check:** Same context/top_k → compare output item_ids (and optionally scores) to baseline. If identical or quality acceptable, keep the skip for speed; otherwise keep current behavior.
- **Speed:** Saves one linear + LN + multiply per forward (small but measurable).

### 2. Beam width

- **Current:** `beam_width = max(4, min(top_k + 2, 20))`.
- **Ablate:** For latency tests, try `beam_width = 1` (greedy). Still gives valid semantic id and one item; top_k would be 1. For top_k > 1 with beam 1 you’d need another strategy (e.g. multiple samples, or accept top_k=1 for that request).
- **Check:** For top_k=1, compare item_id and score to beam_width=12. Quality may drop; measure Hit@k or MRR if available.
- **Speed:** Steps 1–3 batch size 1 instead of 12 → much smaller incremental forwards.

### 3. Sync: skip prefix_tokens transfer when not needed

- **Current:** We always transfer `prefix_tokens` to host and chunk into 4-token lists for trie fallback when `item_id == -1`.
- **Ablate:** If `item_at_leaf` never returns -1 for your trie (all paths resolve to an item), you can skip `backend_transfer(prefix_tokens)` and `to_flat_list` for prefix_tokens when all `item_ids >= 0`.
- **Check:** Ensure in production no item_id is -1 (or accept that in that rare case we return without fallback). Otherwise keep fallback.
- **Speed:** One less device→host transfer and less list building.

### 4. Mask multiply when mask ≡ 1

- **Current:** `apply_aux_encoder` ends with `Nx.multiply(out, mask)`. We pass mask=1.
- **Ablate:** When mask is known to be 1, skip the multiply (or branch in Defn).
- **Check:** Numerically identical.
- **Speed:** Tiny (one less elementwise op per position).

---

## Implemented ablations

- **Aux path:** Set `config :recgpt, :skip_aux_encoder, true` to use `forward_with_cache_no_aux` / `forward_incremental_no_aux` (no aux/mask build or add). Compare item_ids to baseline; if unchanged, keep.
- **Beam width:** Set `RECGPT_BEAM_WIDTH_OVERRIDE=1` (or `config :recgpt, :beam_width_override, 1`) for greedy decode. Compare latency and quality vs adaptive beam.
- **Sync:** Decode now transfers `prefix_tokens` to host only when any `item_id < 0` (trie fallback). When all resolved from `item_at_leaf`, transfer is skipped.
- **Mask skip:** Handled by the no-aux path (no multiply when skip_aux_encoder is true).

## Suggested ablation order

1. **Aux path** — Run with `skip_aux_encoder: true`, trace_predict and a small test set; compare item_ids (and scores if needed). If unchanged, keep skip.
2. **Beam width** — For top_k=1, `RECGPT_BEAM_WIDTH_OVERRIDE=1 mix recgpt.trace_predict` vs default; compare latency and one-item quality.
3. **Sync** — Already on: prefix_tokens transfer skipped when all item_ids ≥ 0. Measure and confirm -1 rarely occurs.
4. **Mask skip** — Use no-aux path (step 1).

---

## How to validate (latency and quality)

- **Latency:** `mix recgpt.trace_predict --runs 10 --jitter-ms 3` (or more runs). Compare total ms, P50/P99, and inference μs between baseline and ablated config. Use the same context/top_k for both.
- **Quality:** `mix recgpt.eval` (or `mix recgpt.eval_grpc`) on the same fixture, checkpoint, and test set. Compare Hit@1, Hit@5, Hit@10, and MRR; ensure ablated config does not regress beyond an acceptable threshold. See [06 Evaluation and testing](06_evaluation_and_testing.md) and `mix recgpt.eval --help`.

---

## What not to remove

- Do **not** remove trie restriction: without it, token sequences can be invalid and not map to any item_id.
- Do **not** remove steps or change 4-token-per-item: semantic id is defined as a 4-token path.
- Do **not** remove WPE or reduce layers without a model change: that would break the semantic.

---

## See also

- [latency_flow.md](latency_flow.md) — E2E flow, GPU tensor graph, where time goes.
- [06_evaluation_and_testing.md](06_evaluation_and_testing.md) — Hit@k, MRR, null hypothesis, eval commands.
