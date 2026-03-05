# Proposal: Dynamic state (trie and beam search)

Sub-proposal of the [documentation index](README.md). Constrained decoding and catalog trie.

---

## Problem or limitation

The decoder must only produce token sequences that match a valid catalog item. Without a trie and catalog-aware beam search, the model could output invalid token sequences that do not map to any item ID.

---

## Proposed improvement

Use a **prefix trie** over fixture token sequences and **catalog-aware beam search** so every top-_k_ recommendation maps to an item ID. The trie is built once at startup; beam search keeps multiple hypotheses and restricts next tokens to valid prefixes.

---

## Trie and beam search in the codebase

- **Trie:** `RecGPT.Trie.build/1` builds a trie from `token_id_list` (4 tokens per item). It supports `lookup/2` (sequence → item_id) and `valid_next_tokens/2` (prefix → valid next tokens).
- **Decode:** `RecGPT.Decode.beam_search_top_k_spmd/8` (beam) and `lookahead_top_k/5` (MTP) take a logits function (`get_logits_4_fn`), trie tensors or item_id_to_tokens, context, and top_k; they return best item_id(s) constrained to the catalog. Strategy is set by `RECGPT_DECODE_STRATEGY` (beam_search or mtp).

The trie is built once at startup and held in `RecGPT.Serve` state. Beam search keeps multiple hypotheses; the trie avoids work on invalid paths.

---

## Future ETS scaling

The trie is currently an **in-memory map**. For very high read concurrency or **live catalog updates** without restart, move it to **Erlang Term Storage (ETS)**:

- ETS table with `read_concurrency: true` and `write_concurrency: true`.
- A **singleton GenServer** owns the table; writes go through it, inference reads directly from ETS in parallel. ETS gives atomic per-object updates, so readers never see torn state. Incremental updates are O(_L_) in sequence length.

The codebase does not use ETS today; this is the recommended path from “load fixture at startup” to “update catalog at runtime” (IURS-style). A future **CatalogService** (live catalog updates) would extend the gRPC API ([recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto)).

---

## Sub-proposals

- **Trie and beam search** (above) — `Trie.build/1`, `lookup/2`, `valid_next_tokens/2`; `Decode.beam_search_top_k_spmd/8`, `lookahead_top_k/5` (MTP).
- **Future ETS scaling** (above) — Optional path for high concurrency and live catalog updates.

---

## See also

- [Documentation index](README.md)
- [15 Layers overview](15_layers_overview.md)
- [20 Layer Recommendation](20_layer_recommendation.md)
- [08 Checkpoint layout](08_recgpt_checkpoint_layout.md)
