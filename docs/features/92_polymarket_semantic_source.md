# Polymarket Stable Semantic Source (RFC 8785 JCS)

Stable, JSON-LD-style semantic ID source text for RecGPT item embeddings, built **only** from Polymarket Gamma API fields. Prioritizes immutable identifiers (conditionId, slug, outcome, category) over editable content (question).

## Problem or limitation

Polymarket item titles (`question + outcome`) can change when markets are edited. Embeddings must be reproducible across runs; volatile titles cause embedding drift. We need a deterministic, stable string for embedding input.

## Proposed improvement

Use a **canonical JSON object** (JSON-LD-style) per RFC 8785 (JCS: JSON Canonicalization Scheme). Same logical content always yields identical byte string regardless of key order or formatting.

### Schema

| Field       | Stability  | Notes                              |
| ----------- | ---------- | ---------------------------------- |
| conditionId | Immutable  | CTF blockchain identifier, hex     |
| slug        | Stable     | URL path, rarely changes            |
| outcome     | Stable     | Yes/No for binary; outcome index   |
| category    | Stable     | Taxonomy (e.g. US-current-affairs) |
| question    | Optional   | Included for embedding richness    |
| tokenId     | Fallback   | When token not in API (placeholder) |

**Required (stable):** conditionId, slug, outcome, category (or empty string if absent).

**Excluded:** description (long, variable; less stable), prices, dates, URLs.

### Canonicalization (required)

`embedding_text` must always be produced via `Jcs.encode/1` (RFC 8785). Never use `Jason.encode!` for embedding source text. Same logical content must yield identical byte string. The API-miss placeholder must also pass through JCS.

### Data flow

1. **PolymarketAPI** — `build_token_to_metadata_map` fetches from Gamma API; `build_stable_semantic_source` produces JCS string.
2. **ConvertJonBecker** — Emits `title` (human-readable) and `embedding_text` (JCS canonical).
3. **Sync** — Writes `items` (title) and `item_embedding_texts` (embedding_text).
4. **FixtureBuild / PretrainRunner** — Use `embedding_text || title` for embedding input when loading from DB.

### Cache

- Path: `out_dir/token_to_title_cache.json`
- Content: `{"token_to_metadata": {token_id => {...}}, "num_tokens": n}`
- Legacy caches with `token_to_title` are converted to minimal metadata (empty stable fields).

### Fallback for API misses

When token not found in API: minimal placeholder passed through `Jcs.encode/1`:

```json
{"category":"","conditionId":"","outcome":"","slug":"","tokenId":"<asset_id>"}
```

When API fails entirely: strict (raise) or lenient (placeholder). Recommend strict.

## See also

- [80 Prediction market trading system](80_prediction_market_trading_system.md)
- [91 FuXi-Linear real timestamps](91_fuxi_linear_real_timestamps.md)
- [ETNF database design](etnf_database_design.md) — `item_embedding_texts` table
