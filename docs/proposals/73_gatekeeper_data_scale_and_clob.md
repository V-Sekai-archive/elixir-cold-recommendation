# Gatekeeper: Data Scaling Performance and Live CLOB Requirements

Can we estimate the Gatekeeper's possible scaling performance with data? Do we have enough? How much do we need from the live CLOB?

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md). Builds on [72 RecGPT 9% → Qwen 75% gap](72_recgpt_9pct_to_qwen_75pct_gap.md), [71 Polymarket dataset scale](71_polymarket_dataset_scale.md), [38 Qwen LoRA](38_sniper_qwen_lora.md).

---

## Problem or limitation

The Qwen Gatekeeper is trained via GRPO on scenarios `(tape_jsonld, xmp_is_gamed, xmp_is_win, outcome_id)`. Each scenario requires:

1. **Labels:** IsGamed, Resolved_Win — from resolved market metadata
2. **The Tape:** Trade legs (price, volume, agent, timestamp) — from trade/orderbook history
3. **Scout pick:** outcome_id — we can simulate from RecGPT or use historical leader picks

We need to know: (a) how Gatekeeper performance scales with scenario count, (b) whether our dataset is sufficient, and (c) how much live CLOB data we need at inference.

---

## Proposed improvement: Scaling estimate and requirements

### 1. Gatekeeper scaling with data (rule-of-thumb)

The Gatekeeper learns a **binary filter** (PICK_ID vs PICK_0), not a generative model. For such tasks, learning curves typically follow a power law: performance improves with `N^α` for some α < 1, then plateaus.

| Scenario count (N) | Expected regime     | Trap Escape (target) | Organic Strike (target) |
| ------------------ | ------------------- | -------------------- | ----------------------- |
| **1k–5k**          | Minimum viable      | 60–70%               | 50–60%                  |
| **10k–30k**        | Good separation     | 75–85%               | 65–75%                  |
| **50k–100k**       | Saturation          | 85–92%               | 70–80%                  |
| **>100k**          | Diminishing returns | ~90%+                | ~75%+                   |

**Formula (heuristic):** No closed-form; empirical. Assume Trap Escape ≈ `70 + 15·log10(N/1e3)` (capped at 95%) and Organic Strike ≈ `55 + 10·log10(N/1e3)` (capped at 80%) for N ∈ [1k, 100k]. At N=100k: Trap Escape ~88%, Organic Strike ~75%.

**Class balance matters:** If only 5% of markets are traps, we need enough trap examples (500+ for 10k total) to learn veto. At ~10–30% trap rate, 10k scenarios → ~1k–3k traps, which is sufficient.

---

### 2. Do we have enough data?

From [71 Polymarket dataset scale](71_polymarket_dataset_scale.md):

| Requirement                  | Available                                                                                                     | Sufficiency                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Resolved markets**         | ~100k–150k (events + markets with resolution)                                                                 | ✓ Enough for 100k scenarios                                                                                                 |
| **Trade history (The Tape)** | Tens–hundreds of millions of OrderFilled; Jon-Becker 36 GiB Parquet                                           | ✓ Can reconstruct Tapes for resolved markets                                                                                |
| **Order-book snapshots**     | Kaggle markets.csv: best bid/ask, spreads, liquidity; Jon-Becker blocks; per-market book (sandeepkumarfromin) | ✓ For microstructure / Tape                                                                                                 |
| **IsGamed labels**           | Must be derived or annotated                                                                                  | ⚠ Optional: use profit-based reward ([§2b](#2b-fallback-no-isgamed--reward-from-resolved_win--profit-only)) if unobservable |

**Scenario construction:** For each resolved market, take trade legs (and optionally order-book state) at T−Δt before resolution (e.g., Δt = 1h, 10min, 1min). One resolved market → multiple scenarios if we vary Δt. So 100k resolved → 100k–300k possible scenarios (1–3 time-slices per market).

**Verdict:** We have **enough** historical data to train the Gatekeeper at the 50k–100k scenario scale. If IsGamed is unobservable, use profit-based reward ([§2b](#2b-fallback-no-isgamed--reward-from-resolved_win--profit-only)); otherwise label quality is the bottleneck.

---

### 2b. Fallback: No IsGamed — reward from Resolved_Win + Profit only

If we **cannot** determine IsGamed (only observe whether the resolved outcome matched our pick and whether we would have made profit), we can still train the Gatekeeper using a **profit-based reward**.

**Observable signals (no IsGamed):**

- **Resolved_Win** = Did Scout's pick match the resolved outcome? (from resolution data)
- **Profit** = Would we have made money? `payoff - cost` (from prices at T−Δt and resolution)

**Profit-only reward:**

| Action  | Resolved_Win | Profit | Reward                        |
| ------- | ------------ | ------ | ----------------------------- |
| PICK_ID | True         | > 0    | +1 (good strike)              |
| PICK_ID | True         | ≤ 0    | 0 (won but thin/no edge)      |
| PICK_ID | False        | —      | -2 or -3 (we would have lost) |
| PICK_0  | True         | —      | -1 (we vetoed a winner)       |
| PICK_0  | False        | —      | +2 (we avoided a loser)       |

We do not distinguish trap vs organic loss; both are "we would have lost." The Gatekeeper learns: **approve when we'd profit, veto when we'd lose**. No IsGamed label required.

**Trade-off:** Profit-based reward is **weaker** than IsGamed because we lose the asymmetric trap signal (-5 hit vs +2 escape). Traps and organic losses are conflated. But it is **always computable** from resolution + Tape (prices → cost, outcome → payoff). If we have Resolved_Win and can compute profit, we have enough to train.

**Scenario construction:** For each resolved market, we know: (1) outcome that won, (2) Scout's simulated pick, (3) prices at T−Δt. So Resolved_Win and Profit are derived without any gamed-ness annotation.

**If IsGamed is never knowable:** Then profit-based reward is the **only** path. We drop Trap Escape Rate and Organic Strike Rate as separate metrics (we cannot measure them). The Gatekeeper learns a **profitability filter**: approve when Tape + Scout pick suggest profit > 0, veto when they suggest loss. We still bridge 9%→75% if the approved subsample has high win rate and positive expectancy—we just never know whether a veto was "trap escape" or "avoided organic loss." Both are good. Design for profit-only from the start; no IsGamed pipeline required.

---

### 3. Live CLOB requirements (inference)

At inference, the Gatekeeper sees a Tape built from **current** trade/orderbook state for Scout's top-1 candidate.

| Dimension       | Requirement                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| **Scope**       | One market (Scout's top-1) — or small set if we extend to top-k                                                |
| **Data needed** | Trade legs (recent fills) + order book (best bid/ask, depth) for the condition_id / outcome_ids in that market |
| **Staleness**   | <1 min for Catalyst (2–10 s edge); <10 min for Combinatorial (10-min window)                                   |
| **Volume**      | ~1–10 KB per request (one market: a few dozen trade legs + order book snapshot)                                |
| **API**         | Polymarket CLOB API or Gamma API; WebSocket for streaming or REST for poll                                     |

**Per-request flow:**

1. Scout returns top-1 item_id (outcome_id)
2. Resolve item_id → (market_id, condition_id, outcome_ids)
3. Fetch CLOB: recent trades + order book for that condition
4. Build Tape (JSON-LD)
5. Gatekeeper inference (PICK_ID or PICK_0)

**Throughput:** One Scout call → one Gatekeeper call → one CLOB fetch. At 1 req/s, CLOB load is trivial (~10 KB/s). Even at 10 req/s, <100 KB/s.

---

### 4. Constraining diff: Training vs inference data

| Phase         | Data source                                      | Volume                             | Latency constraint                                   |
| ------------- | ------------------------------------------------ | ---------------------------------- | ---------------------------------------------------- |
| **Training**  | Historical (Jon-Becker, Kaggle, warproxxx, Dune) | 36–61 GiB; ~100k resolved + trades | None (offline)                                       |
| **Inference** | Live CLOB (Polymarket API)                       | ~1–10 KB per request               | <1 min staleness (Catalyst); <10 min (Combinatorial) |

**Training:** We have enough. Build scenarios from historical resolved + trades. Use profit-based reward if IsGamed is unobservable.

**Inference:** We need a CLOB client that returns, for a given condition_id:

- Recent OrderFilled (or equivalent trade events): last N fills, or last T minutes
- Order book: best bid, best ask, optional depth (e.g., 5 levels)

Polymarket exposes this via CLOB and Gamma APIs. No bulk historical pull needed at inference—only point queries for Scout's candidate(s).

---

### 5. Minimum viable CLOB at inference

| Component      | Minimum                                                      |
| -------------- | ------------------------------------------------------------ |
| **Trade legs** | Last 50–200 fills per outcome (or last 1–10 min of activity) |
| **Order book** | Best bid, best ask; depth optional for Tape richness         |
| **Refresh**    | Per-request (no caching) or cache <30 s for Combinatorial    |

If the market has thin activity, fewer legs are fine; the Gatekeeper learns from variable Tape length. Empty Tape → abstain (PICK_0) is safe default.

---

## Summary

| Question                                 | Answer                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Can we calculate Gatekeeper scaling?** | Heuristic yes: Trap Escape ~70+15·log10(N/1e3)%, Organic Strike ~55+10·log10(N/1e3)% for N ∈ [1k,100k]. No closed form.         |
| **Do we have enough data?**              | Yes. ~100k resolved + trades. Enough for 50k–100k scenarios. Use profit-based reward if IsGamed unobservable.                   |
| **How much live CLOB?**                  | ~1–10 KB per request (one market: recent trades + order book). Sub-minute staleness for Catalyst; sub-10-min for Combinatorial. |

---

## See also

- [38 Qwen LoRA](38_sniper_qwen_lora.md) — GRPO scenarios
- [36 Schema](36_sniper_schema.md) — Tape JSON-LD structure
- [71 Polymarket dataset scale](71_polymarket_dataset_scale.md)
- [72 RecGPT 9% → Qwen 75% gap](72_recgpt_9pct_to_qwen_75pct_gap.md)
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) — Catalyst vs Combinatorial
