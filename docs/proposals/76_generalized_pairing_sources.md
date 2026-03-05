# Generalized Pairing: Sources for Combinatorial / Implied Pairs

Generalize pairing beyond sequence-order: document production-ready sources for combinatorial dependent and implied pairs, and how to build a full historical pairing pipeline. Since this is **historical** data for backtesting, we do **not** require zeroshot inference—we can run LLM, embeddings, and rule-based detectors offline.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md). Builds on [60 Rope bridge](60_rope_bridge_market_analytics_plan.md), [75 Implication graph](75_implication_graph_ycsb_smart_money.md), [71 Polymarket dataset scale](71_polymarket_dataset_scale.md).

---

## 1. Pairing in context

**Pairing** = identifying two or more markets (conditions/outcomes) where resolution of one implies or constrains another. Examples:

| Pair type                        | Example                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| **Binary complement**            | "Kamala wins 2024" + "Trump wins 2024" → sum to 1                    |
| **Win + margin**                 | "GOP wins Georgia by 0%–1.0%" ⇒ "GOP wins Georgia" (binary)          |
| **Popular vote + EC**            | "Democrat wins popular vote" + "Winning candidate wins popular vote" |
| **Margin buckets**               | "GOP wins by 215+" + overall control / popular-vote markets          |
| **Same-condition multi-outcome** | 3+ buckets in one NegRisk condition (butterfly wings)                |

**Why generalize:** Our implication graph (doc 75, 60) builds edges from **leader sequence co-occurrence**. That is one signal. The academic and production world also uses **LLM-driven dependency detection**, **title-embedding cosine similarity**, and **keyword rules**. We can combine sources: use arXiv-enumerated pairs as gold labels, dhruv575 CSV as seed data, and our sequence graph as an additional signal.

**Historical = no zeroshot:** We are backtesting on Apr 2024–Apr 2025 (and similar windows). No live inference required. Run LLM prompts, embeddings, and validation offline. Latency budget applies only when we go live; for pairing _discovery_, we care about recall and precision.

---

## 2. Primary source: arXiv gold standard

### 2.1 Paper

**arXiv:2508.03474** — "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" (Aug 2025, IMDEA Networks)

**Coverage:**

- Full on-chain historical bid/order-filled data, Apr 2024 – Apr 2025
- 86 million bids
- 8,659 single-condition + 1,578 multi-condition/NegRisk markets
- 17,218 total conditions

**Methodology:**

- LLM-driven detection: DeepSeek-R1-Distill-Qwen-32B + Linq-Embed-Mistral embeddings + manual validation
- 13 valid combinatorial dependent market pairs (reduced from 1,576 LLM candidates)
- Full methodology + LLM prompts in appendices (replicable in &lt;200 lines of Python)

**Exact political examples (Tables 7–9, Appendix F):**

- Popular vote + Electoral College winner
- Presidential margin buckets (e.g. "GOP wins by 215+") paired with overall control / popular-vote markets
- State-level win + margin: "Will Democratic candidate win Georgia by 0%–1.0%?" ⇒ "Will Democrat win Georgia?"; same for GOP margins in GA/NC/WI

**Realized arbitrage:** 5 pairs executed for ~$95k total; one pair alone $60k+.

**Access:** Free PDF/HTML on [arXiv](https://arxiv.org/abs/2508.03474). The 13 pairs are explicitly enumerated. No raw CSV of 86M bids; you can replicate detection on top of warproxxx, Jon-Becker, Kaggle, or Dune bulk trade datasets.

**Why start here:** Industry reference for combinatorial strategy design (including the $40M realized arb cited in latency-ceiling docs). Architects can implement the exact dependency graph for binary + scalar margin pairs.

---

## 3. Ready-to-download CSV: election pairs

### 3.1 GitHub: dhruv575/electionFetchingCode

**What it is:** Python pipeline pulling from Polymarket Gamma API; processes historical prices; explicitly pairs Democrat vs Republican markets on identical outcomes.

**Output:**

- `collated_elections.csv` — 79 markets total, 76 perfectly paired, 3 singles
- `senate_collated.csv`

**Pairing logic (implied probability):**

```python
prob = (dem_yes_prob + (1 - rep_yes_prob)) / 2
```

Normalizes complementary binaries. Extensible to scalar margins by joining on event slug or title similarity.

**Coverage:** Full 2024 US election cycle (~$3.2B volume).

**Relevance:** Immediate starting point for state-level win + margin pairing. Script groups by event URL and party keywords; add cosine-similarity or LLM step on titles for GOP-margin + GOP-win pairs.

---

## 4. Cross-platform paired reference (validation)

### 4.1 GitHub: wrongshot/manifold-polymarket-pairs

**What it is:** Single CSV (`polymarket_manifold_pairs.csv.zip`) of 147 paired markets with identical resolution criteria.

**Coverage:** 2023–2025.

**Limitation:** Cross-platform (Polymarket + Manifold), not pure intra-Polymarket combinatorial. Valuable for backtesting implied probabilities on politics/election themes.

---

## 5. How to build your own full historical pairing DB

**Architect recommendation:** Run on any bulk historical dataset (warproxxx S3, Jon-Becker 36 GiB Parquet, Kaggle ismetsemedov, or Dune `polymarket_polygon.market_trades`):

| Step | Action                                                                                                                                                      |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Filter politics/election markets by topic and end-date (as in arXiv paper)                                                                                  |
| 2    | Run arXiv LLM prompt (copy from Appendix B) or simpler: title-embedding cosine similarity + keyword rules ("win by", "margin", "popular vote", state names) |
| 3    | Output: `paired_contracts` table with `market_id_1`, `market_id_2`, `dependency_type`, `first_seen`, `last_seen`, `max_arbitrage_bps`                       |

**Production bots that cite arXiv:2508.03474:**

- alsk1992/CloddsBot — combinatorial detection
- ImMike/polymarket-arbitrage — bundle detection

---

## 6. Summary table (whiteboard reference)

| Source                                  | Type                  | # Pairs / Markets                | Coverage                     | Formats                           | GOP Margin + Win? | Download / Replicable?                  |
| --------------------------------------- | --------------------- | -------------------------------- | ---------------------------- | --------------------------------- | ----------------- | --------------------------------------- |
| **arXiv:2508.03474**                    | Academic + enumerated | 13 combinatorial dependent pairs | Apr 2024–Apr 2025 (86M bids) | Tables in paper                   | Yes               | Free PDF; replicate on any bulk dataset |
| **dhruv575/electionFetchingCode**       | Ready CSV pipeline    | 79 markets (76 paired)           | 2024 US elections            | collated_elections.csv            | Yes               | Direct CSV                              |
| **wrongshot/manifold-polymarket-pairs** | Cross-platform CSV    | 147 pairs                        | 2023–2025                    | polymarket_manifold_pairs.csv.zip | Partial           | Direct CSV                              |
| **Bulk dataset + arXiv LLM code**       | Your own DB           | Unlimited                        | Full history                 | Custom Parquet/DB                 | Yes               | Fully reproducible                      |

---

## 7. Generalizing our pairing strategy

We have **three** pairing signals we can combine:

| Signal                         | Source                                    | Role                                                   |
| ------------------------------ | ----------------------------------------- | ------------------------------------------------------ |
| **Sequence co-occurrence**     | Leader trade legs; RecGPT train_sequences | Adjacent (a, b) in sequences → candidate edge          |
| **LLM / embedding dependency** | arXiv methodology; dhruv575 + cosine sim  | Logical/structural dependency (win+margin, popular+EC) |
| **Enumerated gold pairs**      | arXiv Tables 7–9; dhruv575 CSV            | Validation; seed graph; backtest calibration           |

**Pipeline:**

1. **Seed** — Load arXiv 13 pairs + dhruv575 76 paired markets into `paired_contracts`.
2. **Extend** — Run sequence extraction (mix recgpt.build_implication_graph) on leader data; merge candidate edges.
3. **Filter** — LLM or embedding filter to prune false positives; timeliness (resolution overlap).
4. **Validate** — Backtest: did (outcome_A, outcome_B) ever yield profit? Drop edges that never do.
5. **Serve** — Load final graph into ETS; Scout→butterfly uses it for combinatorial legs.

**No zeroshot for pairing discovery:** All of the above runs offline on historical data. Zeroshot applies to live Scout inference (pretrained model, no Polymarket-specific finetune). Pairing is a **precomputed** dependency graph.

---

## 8. Latency budget note

For **live** combinatorial arb (opportunity lifetime ~minutes–1 h, binding constraint = opportunity cap, target P99 &lt;10 s):

- Drop the arXiv tables (market IDs + dependency graph) into the latency budget spreadsheet.
- Simulate end-to-end capture rates: Scout (~250 ms) + catalog expansion + graph lookup + price fetch + solver.
- The 13 pairs + dhruv575 CSV give concrete market IDs for simulation.

---

## See also

- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — §5.4 implication graph; Scout→butterfly
- [75 Implication graph](75_implication_graph_ycsb_smart_money.md) — YCSB, build, smart money
- [71 Polymarket dataset scale](71_polymarket_dataset_scale.md) — warproxxx, Jon-Becker, Kaggle, Dune
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) — Combinatorial cap &lt;10 s
- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — arXiv primary source
