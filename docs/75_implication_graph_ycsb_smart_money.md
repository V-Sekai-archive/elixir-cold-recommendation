# Implication Graph: YCSB Requirements, Build Guide, and Smart Money

YCSB storage requirements for the implication graph and Gatekeeper pipeline; how to build the implication graph from leader sequences; and frontier-trader ("smart money") research for sequence sourcing.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md). Builds on [31 YCSB storage classification](31_ycsb_storage_classification.md), [60 Rope bridge §5.4](60_rope_bridge_market_analytics_plan.md#54-cross-market-implication-graph-technical-design), [68 Wallet longitudinal](68_wallet_longitudinal_adversary_research.md), [69 Sniper longitudinal leaders](69_sniper_longitudinal_leaders.md).

---

## 1. Components and roles

| Component             | Role                                                                             | Latency       |
| --------------------- | -------------------------------------------------------------------------------- | ------------- |
| **RecGPT Scout**      | Top-1 recommendation from leader sequences; ~9% ceiling alone                    | ~250 ms       |
| **Qwen Gatekeeper**   | Approves or vetoes; learns from profit or trap/win labels. Low; up to 64 tokens. | Low           |
| **Implication graph** | Maps item relationships; built from sequence order (planned). Offline.           | Offline build |

The implication graph feeds Scout→butterfly: it maps `item_id` → `(condition_id, outcome_id)` and stores condition-level edges (A ⇒ B) for cross-market butterflies.

---

## 2. YCSB requirements

Classify storage by access pattern per [31 YCSB](31_ycsb_storage_classification.md). The implication graph and related stores have the following workload types:

### 2.1 Catalog (item_id → condition_id, outcome_id)

| Phase       | Access pattern                                      | YCSB   |
| ----------- | --------------------------------------------------- | ------ |
| Build       | Bulk upsert when syncing from Polymarket            | A-like |
| Graph build | Point read by item_id for each (a, b) in sequences  | C      |
| Serve       | Point read by item_id for Scout→butterfly expansion | C      |

**YCSB:** **B** (read mostly) or **C** (read only) if catalog is static after init.

### 2.2 Implication graph (condition_id → successors)

| Phase      | Access pattern                                                | YCSB                                |
| ---------- | ------------------------------------------------------------- | ----------------------------------- |
| Build      | Stream sequences; accumulate edges in memory; bulk write JSON | F (in-memory) → A-like (bulk write) |
| Serve      | Point read by condition_id (get successors)                   | C                                   |
| Validation | Full-edge scan for backtest / aggregate stats                 | E                                   |

**YCSB:** **C** primary; **E** if validation scans.

### 2.3 Gatekeeper scenarios (training)

| Phase    | Access pattern                                                      | YCSB                         |
| -------- | ------------------------------------------------------------------- | ---------------------------- |
| Build    | Stream resolved markets; construct (tape, xmp, outcome); bulk write | C stream → A-like bulk write |
| Training | Stream batches; read-only                                           | C                            |

**YCSB:** **C** (stream, read-only).

### 2.4 Store fit summary

| Store                 | Catalog      | Implication graph          | Gatekeeper scenarios |
| --------------------- | ------------ | -------------------------- | -------------------- |
| **SQLite (Ecto)**     | ✓ B, C       | ✓ Persistence; load to ETS | ✓ If stored in DB    |
| **ETS**               | ✓ Hot path   | ✓ Hot path; O(1) adjacency | —                    |
| **File**              | ✓ items.json | ✓ implication_graph.json   | ✓ JSON/Parquet       |
| **DuckDB** (columnar) | —            | ✓ Validation scans (E)     | ✓ Analytic queries   |

---

## 3. How to build the implication graph

### 3.1 Prerequisites

1. **Catalog with `condition_id`** — Extend `items.json` or add `item_condition_mapping.json`:
   ```json
   { "id": 0, "title": "...", "condition_id": "0x...", "outcome_id": "0x..." }
   ```
2. **Train sequences** — Lists of item_ids from leader/wallet trade legs. Format: `[[i1, i2, i3, ...], ...]`.
3. **Polymarket metadata** (optional) — For timeliness filter (resolution overlap).

### 3.2 Build steps

| Step                    | Action                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| 1. **Catalog**          | Ensure item_id → condition_id exists (Polymarket fetch or manual mapping).                              |
| 2. **Extract**          | Run `mix recgpt.build_implication_graph`; reads train_sequences, emits candidate edges.                 |
| 3. **Merge** (optional) | Append domain rules from `domain_rules.json` (hand-curated edges).                                      |
| 4. **Filter**           | Apply timeliness (resolution overlap) and topical similarity if metadata/embeddings available.          |
| 5. **Validate**         | Backtest: for each edge, check if (outcome_A, outcome_B) ever yielded profit; drop edges that never do. |

### 3.3 Algorithm: transition extraction

```
Input: sequences = [[i1, i2, i3, ...], ...]  (item_ids)
       catalog: item_id -> condition_id
Output: edge_counts: %{(c_a, c_b) => count}

for each seq in sequences:
  for each adjacent pair (a, b) in seq:  # a at pos i, b at pos i+1
    c_a = catalog[a]
    c_b = catalog[b]
    if c_a != c_b and c_a != nil and c_b != nil:
      edge_counts[(c_a, c_b)] += 1

Filter: keep edges where count >= min_count (e.g. 5)
```

**Source:** Sequence order (item A before item B in leader sequences). Co-occurrence count = weight. No embeddings required for primary build.

### 3.4 Mix task contract

```
mix recgpt.build_implication_graph \
  --train data/polymarket/train_sequences.json \
  --catalog data/polymarket/items.json \
  --min-count 5 \
  --out data/polymarket/implication_graph.json
```

### 3.5 Output schema

```json
{
  "version": 1,
  "edges": [
    { "from": "0x...", "to": "0x...", "weight": 42, "source": "sequence" },
    { "from": "0x...", "to": "0x...", "weight": 12, "source": "domain_rule" }
  ],
  "stats": { "num_edges": 150, "num_conditions": 80 }
}
```

### 3.6 Optional filters

| Filter         | Purpose                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------- |
| **Topical**    | Embed market titles; cosine_sim(A, B) ≥ threshold → keep edge. Prune unrelated co-occurrences. |
| **Timeliness** | Keep edges where resolution_end_a and resolution_end_b overlap within ±7 days.                 |
| **k-step**     | For [A, B, C, D], add (A,D) as well as (A,B), (B,C), (C,D). Weight by 1/(k+1).                 |

### 3.7 Current status

No implementation exists. Build order: extend catalog schema → implement `RecGPT.ImplicationGraph.extract_from_sequences/2` → add mix task → add filters.

---

## 4. Frontier traders ("smart money"): research summary

Public sources (vpn07, PolyTrack, PANews, TradeTheOutcome, Medium) as of early 2026:

### 4.1 Who counts as smart money?

| Criterion         | Typical threshold                 |
| ----------------- | --------------------------------- |
| **Activity**      | Regular trading over months/years |
| **Win rate**      | 55%+ over 100+ resolved markets   |
| **Position size** | $5,000–$500,000+ per bet          |
| **Volume**        | $100,000+ total                   |

**Caveat:** Reported win rates can be inflated by "zombie orders" (unclosed losing positions). One top whale's 73.7% historical win rate drops to 53.3% when accounting for open positions (PANews, 27,000-trade analysis).

### 4.2 Whale strategy archetypes

| Type                  | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| **Quant Hedge**       | Multi-directional automated systems; complex position relationships |
| **Swing Trader**      | Profits from probability fluctuations, not just resolution outcomes |
| **Asymmetric Hedger** | Large YES + small NO hedges; risk-adjusted returns                  |
| **Arb Machine**       | High-frequency, small-edge arbitrage across many markets            |
| **Domain Specialist** | Deep focus on 1–2 categories; expert-level knowledge                |

### 4.3 Wallets to follow vs avoid

**Follow:** Medium-frequency (100 predictions/month); 60%+ win rate over 4+ months; deep liquidity in major markets; niche specialists.

**Avoid:** Bot wallets (spreads not replicable); insider one-off bets; low liquidity; gamblers with low win rates; high-volatility chasers. Only ~12.7% of Polymarket users are profitable.

### 4.4 Tracking tools

| Tool                                   | Role                                                  |
| -------------------------------------- | ----------------------------------------------------- |
| **PolyTrack** (polytrackhq.app)        | Real-time alerts; ROI filtering; 60%+ win-rate whales |
| **Polygon Explorer** (polygonscan.com) | Direct blockchain; wallet→trade legs                  |
| **Polymarket API**                     | Custom tracking via proxy addresses                   |

### 4.5 Implications for implication graph

- **Sequence source:** Leaders' trade legs (wallet→outcome→item_id sequences) from on-chain or CLOB data. Cluster by trajectory shape to separate strategy families (Quant vs Domain Specialist vs Arb).
- **Trap avoidance:** Trap-hitters vs trap-escapers have distinguishable patterns. Feed escape patterns into Gatekeeper GRPO.
- **Don't blind copy:** Use leader data as _context_ for RecGPT, not as copy-trade signal. Focus on combinatorial/catalyst where speed is less critical.

---

## 5. Pipeline integration

| Step                | Implication graph role                                                          |
| ------------------- | ------------------------------------------------------------------------------- |
| **N (catalog)**     | item_id → condition_id; leader IDs; wallet→outcome mappings                     |
| **N+1 (RecGPT)**    | Pretrain on leader sequences; sequences feed graph build                        |
| **N+2 (cluster)**   | Cluster by trajectory shape; strategy taxonomy informs which sequences to use   |
| **N+3 (Qwen LoRA)** | Gatekeeper veto; GRPO informed by trap-hit/escape rates from longitudinal trace |

---

## See also

- [31 YCSB storage classification](31_ycsb_storage_classification.md)
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — §5.4 implication graph design
- [76 Generalized pairing sources](76_generalized_pairing_sources.md) — arXiv, dhruv575, build-your-own pairing DB
- [68 Wallet longitudinal](68_wallet_longitudinal_adversary_research.md)
- [69 Sniper longitudinal leaders](69_sniper_longitudinal_leaders.md)
- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — combinatorial arb; heuristic reduction
