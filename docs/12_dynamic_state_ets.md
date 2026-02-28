# Dynamic State: Constrained Decoding and Catalog Trie

The decoder must only produce token sequences that match a valid catalog item. The system uses a **prefix trie** over fixture token sequences and **catalog-aware beam search** so every top-_k_ recommendation maps to an item ID.

---

## Trie and beam search in the codebase

- **Trie:** `RecGPT.Trie.build/1` builds a trie from `token_id_list` (4 tokens per item). It supports `lookup/2` (sequence → item_id) and `valid_next_tokens/2` (prefix → valid next tokens).
- **Decode:** `RecGPT.Decode.beam_search/4` and `beam_search_top_k/4` take a logits function (from `RecGPT.Inference`), the trie, context token IDs, and beam width; they return best item_id(s) constrained to the catalog.

The trie is built once at startup and held in `RecGPT.Serve` state. Beam search improves over greedy decoding by keeping multiple hypotheses; the trie avoids work on invalid paths.

---

## Optional ETS scaling

The trie is currently an **in-memory map**. For very high read concurrency or **live catalog updates** without restart, move it to **Erlang Term Storage (ETS)**:

- ETS table with `read_concurrency: true` and `write_concurrency: true`.
- A **singleton GenServer** owns the table; writes go through it, inference reads directly from ETS in parallel. ETS gives atomic per-object updates, so readers never see torn state. Incremental updates are O(_L_) in sequence length.

The codebase does not use ETS today; this is the recommended path from “load fixture at startup” to “update catalog at runtime” (IURS-style). A future **CatalogService** (live catalog updates) would extend the gRPC API ([13](13_grpc_api.md)).

**Next:** [13_grpc_api.md](13_grpc_api.md).
