# SPMD decode flow

Sub-proposal of the [documentation index](README.md). Describes the SPMD-style beam search used for next-item recommendation: trie tensors, device-side decode, and single CPU sync. See also: [20 Layer Recommendation](20_layer_recommendation.md), [42 Latency and performance](42_latency_and_performance.md).

---

## Problem or limitation

Catalog-aware beam search over 4 tokens (one RecGPT item) requires:

- Restricting each step to valid prefix tokens in the trie
- Batched inference to avoid per-candidate forward passes
- Minimal CPU–device synchronization to reduce latency

A list-based decode that walks the trie on the CPU and calls the model per candidate forces repeated device syncs (`Nx.to_number`, `Nx.to_list`) and prevents full GPU utilization. The beam search loop becomes a latency bottleneck even when inference itself is batched.

---

## Proposed improvement

**SPMD (Single Program Multiple Data) style:** Keep trie and scoring on the device; perform one sync at the end. The entire decode runs as tensor operations on the inference backend (e.g. EXLA/CUDA) until we need the final top-k item IDs on the host.

### Flow overview

1. **Context:** Context item IDs (e.g. `[0, 1]`) → gather tokens from `item_id_to_tokens` tensor → `context_tokens` shape `{1, context_len}`. Empty context uses a single padding token (Nx disallows zero-sized dimensions).

2. **Step 0:** Forward context only → logits. Mask by `next_state[0, :] >= 0` (valid first tokens). Top-k by score → `top_token_ids`, `new_state_ids`, `prefix_tokens` (1 column).

3. **Steps 1–2:** For each beam, build batch `[context || prefix]`, forward → logits. Valid mask from `next_state[state_ids, :]` (use `state_ids[batch_indices]`, not beam indices, as row into trie). Top-k → update `state_ids`, `prefix_tokens`, `beam_scores`.

4. **Step 3:** Same pattern; valid mask from `item_at_leaf[state_ids, :]`. Result is `item_ids` (one per beam candidate).

5. **Single sync:** Transfer `item_ids`, `beam_scores`, `prefix_tokens` to host. Resolve any `-1` via `Trie.lookup(trie, prefix_tokens)` when tensor `item_at_leaf` yields no match. Filter, sort, dedupe, take top_k.

### Modules and responsibilities

| Module            | Responsibility                                                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT.Trie**   | Build map trie from `token_id_list`; `to_tensors/2` exports `next_state` and `item_at_leaf` tensors for device-side lookup.                                             |
| **RecGPT.Decode** | `beam_search_top_k_spmd/7`: accepts trie tensors, `item_id_to_tokens`, context item IDs, `batch_tensor_fn`, backend; returns `{:ok, [item_id]}` or `:not_found`.        |
| **RecGPT.Serve**  | Loads state with `trie_tensors`, `item_id_to_tokens_tensor`, `get_logits_batch_tensor_fn`; `recommend/3` calls SPMD exclusively. Raises if SPMD components are missing. |

### Trie tensor layout

- **`next_state`** `{num_states, vocab_size}` (s32): `next_state[state_id, token_id]` = next state ID (depth 0–2) or -1.
- **`item_at_leaf`** `{num_states, vocab_size}` (s32): `item_at_leaf[state_id, token_id]` = item ID when depth=3 and token completes path, else -1.
- State IDs are assigned by BFS in `Trie.collect_trie_transitions/2`; root = 0.

### Edge cases

- **Empty context:** Use single padding token `[0]`; stub/model must handle `seq_len = 1`.
- **Valid pairs < beam_width:** Top-k can pick invalid (state, token) pairs → `new_state_ids` may be -1. Clamp to 0 before gather to avoid out-of-bounds; invalid beams are masked out in later steps.
- **`item_at_leaf` returns -1:** Fall back to `Trie.lookup(trie, prefix_tokens)` on the host when trie map is provided.

---

## Limitations and future work

| Limitation                                                              | Impact                                                                    | Possible improvement                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Single sync still transfers beam_width × (item_ids + scores + 4 tokens) | Minor host overhead; acceptable for beam_width ≤ 20                       | Consider keeping scores on device for downstream ranking                                |
| Empty context uses dummy token                                          | Stub/tests must handle `[0]`; model may behave differently for true empty | Use a dedicated BOS token or model change for empty-context semantics                   |
| No true batched multi-context SPMD                                      | `recommend_batch` calls `recommend` per context                           | Single batch over B contexts with beam_width each (larger batch, more complex indexing) |

---

## See also

- [20 Layer Recommendation](20_layer_recommendation.md) — Serve, Decode, Trie roles.
- [42 Latency and performance](42_latency_and_performance.md) — Batched inference, KV-cache, backend.
- [04 RecGPT library](04_recgpt_library.md) — Module reference.
- `lib/recgpt/trie.ex` — `to_tensors/2`, `collect_trie_transitions/2`.
- `lib/recgpt/decode.ex` — `beam_search_top_k_spmd/7`, `spmd_step/13`, `gather_2d/3`.
- `lib/recgpt/serve.ex` — `load_state/3`, `recommend/3`, `build_get_logits_4_fn/2`.
- `test/recgpt/decode_spmd_test.exs` — Trie tensor verification, SPMD behavior tests.
