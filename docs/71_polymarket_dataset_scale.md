# Polymarket Dataset Scale: Total Possible Size

Calculation of the total possible scale across public Polymarket data sources. Sources and overlap noted.

---

## Source-by-source scale

| Source                                     | Format              | Size (explicit)                                                    | Rows / records                              | Time range          |
| ------------------------------------------ | ------------------- | ------------------------------------------------------------------ | ------------------------------------------- | ------------------- |
| **Kaggle: ismetsemedov**                   | CSV                 | events: 100 MB, markets: 202 MB                                    | 43,840 + 100,795                            | Jul 2022 – Dec 2025 |
| **warproxxx/poly_data**                    | CSV (.xz)           | events + markets ≈ 302 MB; orderFilled_complete.csv.xz **unknown** | Same events/markets; raw order-filled large | Launch – present    |
| **Jon-Becker/prediction-market-analysis**  | Parquet             | **36 GiB** compressed                                              | Full markets + trades + blocks              | Full historical     |
| **Kaggle: sandeepkumarfromin**             | Per-market CSV/JSON | **Unknown** (3,385 markets × 4 file types)                         | Per-market: book, holder, price, trade      | Multi-year          |
| **Dune: polymarket_polygon.market_trades** | SQL → CSV export    | Query-dependent; export limits apply                               | Full on-chain OrderFilled                   | 2022 – present      |

---

## Explicit size sum

| Component                                 | Size       |
| ----------------------------------------- | ---------- |
| Kaggle ismetsemedov (events + markets)    | 302 MB     |
| warproxxx (events + markets, overlapping) | 302 MB     |
| warproxxx orderFilled_complete.csv.xz     | Unknown    |
| Jon-Becker full archive                   | **36 GiB** |
| Kaggle sandeepkumarfromin                 | Unknown    |
| Dune export                               | Unknown    |

**Known total:** 302 MB + 36 GiB ≈ **37.3 GiB** (assuming we count Kaggle + Jon-Becker as non-overlapping snapshots).

---

## Estimated scale (orderFilled + per-market)

**orderFilled_complete.csv.xz (warproxxx):**

- "Saves >2 days of scraping" ⇒ large
- Full historical order-filled events; compressed .xz typical ratio ~3–5×
- Goldsky subgraph: all OrderFilled events since Polymarket launch
- Rough estimate: if Jon-Becker trades are similar but Parquet-compressed, and Jon-Becker total is 36 GiB with markets + trades + blocks, trades alone might be ~20–30 GiB equivalent in raw CSV. So orderFilled_complete could be **5–15 GiB** compressed if it's a full dump. **Conservative: 2–10 GiB**.

**Kaggle sandeepkumarfromin (3,385 markets):**

- 4 files per market (book, holder, price, trade)
- 13,540+ files. Assume 20–100 KB avg per file → **270 MB – 1.4 GB**.
- If many markets are active: up to **2–5 GB**.

**Dune export:**

- Full table export limited by plan; bulk could reach **1–10+ GiB** uncompressed depending on filters.

---

## Total possible scale (upper bound)

| Component                      | Low estimate | High estimate |
| ------------------------------ | ------------ | ------------- |
| Kaggle ismetsemedov            | 302 MB       | 302 MB        |
| warproxxx (events+markets)     | 302 MB       | 302 MB        |
| warproxxx orderFilled          | 2 GiB        | 10 GiB        |
| Jon-Becker                     | 36 GiB       | 36 GiB        |
| Kaggle sandeepkumarfromin      | 270 MB       | 5 GB          |
| Dune export                    | 0            | 10 GiB        |
| **Total (additive, no dedup)** | **~39 GiB**  | **~61 GiB**   |

**With deduplication (overlap):** events/markets appear in multiple places. Jon-Becker's 36 GiB is described as "largest publicly available combined" — likely subsumes much of the others. Realistic **unique data scale:** **~36–45 GiB** (Jon-Becker + incremental per-market Gamma + any Dune-specific views).

---

## Record-count scale

| Source              | Rows (explicit)                                           |
| ------------------- | --------------------------------------------------------- |
| Kaggle ismetsemedov | 144,635 (43,840 + 100,795)                                |
| Jon-Becker          | Not given; Parquet with full trades                       |
| warproxxx           | Same events/markets + full order-filled (likely millions) |
| sandeepkumarfromin  | 3,385 markets × variable trades per market                |
| Dune                | Full on-chain OrderFilled (millions of rows)              |

**Rough order of magnitude:** Polymarket since 2022 has hundreds of thousands of markets/events and **tens to hundreds of millions** of trade events. Jon-Becker's 36 GiB Parquet with trades + blocks supports that scale.

---

## Summary

| Metric                            | Value                                        |
| --------------------------------- | -------------------------------------------- |
| **Explicit total (no overlap)**   | ~37.3 GiB                                    |
| **Estimated total (all sources)** | **39–61 GiB**                                |
| **Realistic unique scale**        | **36–45 GiB**                                |
| **Largest single archive**        | Jon-Becker: 36 GiB                           |
| **Market/event rows**             | ~100k–150k (from Kaggle); more in Jon-Becker |
| **Trade rows**                    | Tens to hundreds of millions (full on-chain) |

---

## Relation to RL scaling constraints

[70 RL scaling constraining diff](70_rl_scaling_constraining_diff.md) establishes that RL-scaling needs **~2× orders of magnitude** more compute than inference-scaling for the same capability gain (10× RL ≈ 3× inference; 10,000× RL ≈ 100× inference). Dataset size adds another constraint.

### Reward-signal budget vs RL scaling

| Dimension          | Polymarket dataset                                                 | RL scaling (Ord)                                                        |
| ------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Scale**          | ~36–61 GiB; ~100k markets/events; tens–hundreds of millions trades | 10,000× RL compute for 20%→80% gain                                     |
| **Reward density** | Resolved markets = sparse verifiable rewards (~100k over 2.5y)     | RL receives <1/10,000 as much info per FLOP vs pre-training             |
| **Ceiling**        | Data is fixed; cannot synthesize more resolved outcomes            | More compute + same data → diminishing returns (reusing sparse rewards) |

**Implication:** Polymarket gives a **finite reward budget**. RL scaling says we need 10× RL compute to match a 3× inference boost—but we can’t 10× the number of resolved markets without waiting for real-world resolution. We are **data-limited before compute-limited**. Frontier labs (o1, o3) use synthetic reasoning (MCTS, self-play) for unlimited signal; we have real resolved outcomes only. The dataset size caps how far RL can take us even if compute were free.

### Constraining diff: dataset vs scaling regime

| Regime                    | Dataset dependency                                  | Polymarket constraint                              |
| ------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| Pre-training / supervised | Needs tokens; scales with data                      | ~40 GiB supports pretrain; not the bottleneck      |
| RL-scaling                | Needs reward signal; **fixed by resolved outcomes** | ~100k resolved markets = hard ceiling on RL signal |
| Inference-scaling         | No training data consumed                           | Dataset size irrelevant; latency/cost only         |

**Takeaway:** For Polymarket-driven RL (e.g. GRPO on resolved outcomes per [38 Sniper Qwen LoRA](38_sniper_qwen_lora.md)), the ~40 GiB archive provides ample _context_ (trades, orderbook, microstructure) but the _reward_ budget is the count of resolved markets. RL scaling constraints + fixed reward budget ⇒ prioritize inference-scaling and rules over pure RL if data-limited.

---

## See also

- [70 RL scaling constraining diff](70_rl_scaling_constraining_diff.md)
- [73 Gatekeeper data scale and CLOB](73_gatekeeper_data_scale_and_clob.md) — Gatekeeper data sufficiency; live CLOB
- [50 Virix Polymarket strategy review](50_virix_polymarket_strategy_review.md)
- [60 Rope bridge market analytics](60_rope_bridge_market_analytics_plan.md)
- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
