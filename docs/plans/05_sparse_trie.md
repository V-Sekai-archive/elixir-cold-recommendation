# Plan: STATIC-style sparse trie

**Profit:** 1.3 | **Effort:** High | **Gain:** Critical for 13M

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

Replace dense `{num_states, vocab_size}` trie with sparse representation so 13M-item catalogs fit in memory.

---

## Problem

Dense trie at 13M items: `num_states` can reach 10M+; `num_states × 15361 × 4 bytes` ≈ **hundreds of GB**. Infeasible.

---

## Approach

Store only valid `(state, token) → next_state/item_id` in CSR or similar. Memory: O(num_states × avg_branching) ≈ **low single-digit GB**.

---

## Changes

- [lib/recgpt/trie.ex](../lib/recgpt/trie.ex): Add `to_sparse_tensors/2` returning CSR: `{row_offsets, col_indices, values}` for `next_state` and `item_at_leaf`
- [lib/recgpt/decode.ex](../lib/recgpt/decode.ex): Replace `Nx.gather(next_state, row_indices)` with sparse lookup. For each `(state_id, token_id)` pair, binary search or index into CSR to get `next_state_id` or `item_id`. May require batched CPU lookup or custom Nx/EXLA sparse ops.
- Nx has no native sparse; options: (a) CSR on CPU with batched lookup, (b) packed dense format (e.g. list of valid tokens per state), (c) XLA sparse if exposed by EXLA.

---

## Profile

Memory footprint and decode latency on synthetic 13M trie.
