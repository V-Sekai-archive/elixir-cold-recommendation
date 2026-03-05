# Polymarket Stable Semantic Source (RFC 8785 JCS) — Archived

Stable, JSON-LD-style semantic ID source text for RecGPT item embeddings when using **Jon-Becker Polymarket** (`--format jon_becker`). Built only from Polymarket Gamma API fields. Prioritizes immutable identifiers (conditionId, slug, outcome, category) over editable content (question).

## Summary

- **Canonicalization:** Use `Jcs.encode/1` (RFC 8785) for all `embedding_text`; never `Jason.encode!`.
- **Data flow:** PolymarketAPI → ConvertJonBecker (title + embedding_text) → Sync (items + item_embedding_texts) → FixtureBuild/PretrainRunner use `embedding_text || title`.
- **Cache:** `out_dir/token_to_title_cache.json` with `token_to_metadata`.
- **Fallback:** Minimal placeholder `{"category":"","conditionId":"","outcome":"","slug":"","tokenId":"<asset_id>"}` via JCS when token not in API.

See [93 Pretraining plan](../features/93_pretraining_plan.md) for optional Polymarket pipeline. ETNF: [etnf_database_design](../features/etnf_database_design.md) — `item_embedding_texts` table.
