# Strategy Review: Our Position vs Leading Firms

This document explains our prediction-market strategy, how it differs from leading firms, and when it makes sense to invest more compute in our models. We use our tools (RecGPT Scout, Qwen Gatekeeper, implication graph, rope bridge) and insights from Toby Ord's work on inference-scaling. Builds on [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md), [70 RL scaling](70_rl_scaling_constraining_diff.md), [60 Rope bridge](60_rope_bridge_market_analytics_plan.md).

---

## 1. Our strategy and tools

We run a pipeline that (1) recommends trades from leader patterns, (2) filters bad picks with a learned Gatekeeper, and (3) sizes positions with survivorship rules. Here is the stack:

| Component             | Role                                                            | Latency / cost              |
| --------------------- | --------------------------------------------------------------- | --------------------------- |
| **RecGPT Scout**      | Recommends one outcome from leader sequences; ~9% ceiling alone | ~250 ms                     |
| **Qwen Gatekeeper**   | Approves or vetoes; learns from profit or trap/win labels       | Low; up to 64 tokens        |
| **Implication graph** | Maps item relationships; built from sequence order (planned)    | Offline                     |
| **Rope bridge**       | Kelly sizing, survivorship, butterfly profit; paper-trade first | —                           |
| **Pipeline**          | Catalog → RecGPT → Cluster → Qwen finetune                      | Targets Mid-Tail, Long Tail |

**Where we play:** We focus on **Catalyst** (2–10 second windows) and **Combinatorial** (~10 min–1 hour). We **skip** the fastest regimes (under 100 ms, under 150 ms) because our Scout cannot compete on speed. [61](61_strategy_given_latency_ceiling.md).

---

## 2. What leading firms do

Leading prediction-market firms use several strategies. We compete in some and skip others:

| Strategy                  | Leading firms                                                  | Us                                                      |
| ------------------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| **Ultra-fast arbitrage**  | High-frequency bots; most arb profits; execute in milliseconds | **Skip** — our Scout needs ~250 ms per recommendation   |
| **Market making**         | Quote both sides; post liquidity; high win rates               | Not our focus; we use others' tools for fast regimes    |
| **Combinatorial arb**     | AI to find related markets; prune by timeliness and topic      | **Our fit** — implication graph, sequence order, RecGPT |
| **Information arbitrage** | Top 1%; proprietary data (e.g. polls); predict mispricing      | Gatekeeper veto from Tape; we filter, not predict       |
| **Domain specialization** | Niche expertise; high win on rare high-conviction bets         | Long Tail target; early edge before crowding            |

**Our wedge:** Leading firms compete on speed in the busiest markets. We focus on **Mid-Tail and Long Tail**—windows of seconds to minutes where opportunity lifetime matters more than raw speed. Our edge: recommendation from leader sequences, a learned filter, and an implication graph, in regimes where millisecond bots are irrelevant.

---

## 2b. Differential contrast: leading firms vs us

Side-by-side, the main differences are:

| Dimension           | Leading firms                                                                                      | Us                                                                               | What this means                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Speed**           | Sub-100 ms; dedicated infrastructure; most arb profits                                             | ~250 ms Scout; 2–10 s or ~10 min windows                                         | They win on raw speed. We win where the opportunity lasts long enough that speed is not the bottleneck. |
| **Signal**          | Primary sources (gov, court, regulatory); proprietary polls; beat the market to news by 30 s–3 min | Tape (trade flow, leader activity); the market _is_ our signal; no external news | They front-run news. We filter on what is already in the Tape. Different wedge.                         |
| **Combinatorial**   | AI and heuristics (timeliness, topic); expert-validated                                            | Sequence-order implication graph (planned); RecGPT item co-occurrence            | Same regime; they use semantics, we use leader sequences. We are behind—not built yet.                  |
| **Information arb** | Top 1%; bespoke data; predict from polls                                                           | Gatekeeper veto from Tape; filter, don't predict; no polls                       | They predict outcomes. We reject bad picks. We abstain more often.                                      |
| **Cross-platform**  | Polymarket–Kalshi; 3–5 times a week; sub-10 ms tools; 5%+ spreads                                  | Single platform (Polymarket only)                                                | They arbitrage _between_ venues. We optimize _within_ Polymarket.                                       |
| **Liquidity**       | Quote both sides; post liquidity                                                                   | Take liquidity (Scout → Gatekeeper → trade)                                      | They provide; we take.                                                                                  |
| **Risk and sizing** | Virix: narrow focus, small capital, no formal Kelly                                                | Kelly, Greeks, shock tests, bankruptcy rule                                      | We quantify survivorship; they avoid risk by focusing narrowly.                                         |
| **Trap avoidance**  | No explicit; some copy traps                                                                       | Learned veto; asymmetric reward (penalize trap hits, reward vetoes)              | We learn to veto; they do not. A differentiator.                                                        |
| **Rules**           | Hand-crafted scoring (markets, orderbook, wallets)                                                 | RecGPT + Gatekeeper (learned from sequences and rewards)                         | They write rules; we train on data.                                                                     |
| **Order book**      | Full depth; microstructure                                                                         | Tape plus best bid/ask; depth optional                                           | They use book structure; we use flow. We could add richer microstructure if performance plateaus.       |

**Bottom line:** Leading firms compete on speed and on being first to news. We compete on filter quality (veto traps, size with Kelly) and on regime choice (Catalyst/Combinatorial, where our ~250 ms fits). We filter; we do not predict.

---

## 2c. Contrasting diff: detail and status

| Dimension                 | Leading traders                                                                                           | Us                                                                                                                                    | Status                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Ultra-fast arb**        | Sub-100 ms; dedicated infra; 73% arb profits                                                              | Bypass                                                                                                                                | ✓ Deliberate (latency ceiling)                                        |
| **Market making**         | Two-sided quoting; 78–85% win                                                                             | Other tools for fast regimes                                                                                                          | ✓ Deliberate (not our wedge)                                          |
| **Combinatorial arb**     | AI/heuristics; [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474); expert-validated | Implication graph (planned); not yet built                                                                                            | ⚠ **Gap** — build `mix recgpt.build_implication_graph`                |
| **Information arb**       | Bespoke polls; systematic mispricing; top 1%                                                              | Gatekeeper veto from Tape; filter, not predict                                                                                        | ✓ Deliberate (different wedge)                                        |
| **Domain specialization** | 96% win on rare high-conviction                                                                           | Long Tail target                                                                                                                      | ✓ Match                                                               |
| **Rule-based scoring**    | Hand-crafted (markets, orderbook, wallets; Virix)                                                         | RecGPT + Gatekeeper (learned)                                                                                                         | ✓ Different path                                                      |
| **Order book / CLOB**     | Full depth; microstructure                                                                                | Tape + best bid/ask; depth optional [73](73_gatekeeper_data_scale_and_clob.md)                                                        | ⚠ **Partial** — consider richer microstructure if Gatekeeper plateaus |
| **Wallet tracking**       | On-chain wallet scanner (Virix)                                                                           | Leaders' trade legs; longitudinal tracing [68](68_wallet_longitudinal_adversary_research.md), [69](69_sniper_longitudinal_leaders.md) | ⚠ **Planned** — trajectory clustering not yet in pipeline             |
| **Expert validation**     | Heuristic reduction validated by experts                                                                  | None documented                                                                                                                       | ⚠ **Gap** — consider domain-expert review of implication graph        |
| **Kelly / survivorship**  | Virix: narrow focus, no Kelly                                                                             | Kelly, Greeks, shock tests [60](60_rope_bridge_market_analytics_plan.md)                                                              | ✓ We have it; they do not                                             |
| **Trap veto**             | No explicit (Virix)                                                                                       | Learned veto; asymmetric reward                                                                                                       | ✓ We have it; differentiator                                          |
| **News / signal**         | Perplexity for news (some firms)                                                                          | Tape = signal; news too slow [50](50_virix_polymarket_strategy_review.md)                                                             | ✓ Aligned                                                             |
| **Cross-platform arb**    | Polymarket–Kalshi; 3–5×/week; sub-10 ms (2026)                                                            | Single-platform                                                                                                                       | ✓ Bypass (would need Kalshi API)                                      |

**Gaps to address:**

- **Implication graph** — Not built. Blocking: catalog must include `condition_id`.
- **Wallet longitudinal** — Planned in 68/69; trajectory clustering not yet in pipeline.
- **Expert validation** — Combinatorial papers use it; we do not yet.
- **Order book microstructure** — Optional today; add if Gatekeeper plateaus.
- **Wash trading** — ~20% of volume can be wash (2026); Tape may be noisy. Consider filtering if the Gatekeeper picks up spurious patterns.

---

## 2d. 2026 landscape (from public sources)

Public reports (TradeTheOutcome, QuantVPS, PolyTrack, etc.) as of early 2026:

| Fact                                                                                                                                                                              | Source                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **$21.5B** Polymarket volume in 2025; **80%** of participants net losers                                                                                                          | Forbes, TradeTheOutcome    |
| **Bid-ask spreads** tightened from 4.5% (2023) to 1.2% (2025); mechanical arbitrage largely dead                                                                                  | TradeTheOutcome            |
| **$40M** in arb profits (Apr 2024–Apr 2025); bots now close gaps in milliseconds                                                                                                  | TradeTheOutcome, PolyTrack |
| **Top 1%** strategy = **information arbitrage** (35–95% annual returns)                                                                                                           | Multiple                   |
| **Information arb edge**: Polymarket lags **30 s–3 min** on breaking news; first 15 s = primary sources (gov, court, regulatory); 15–60 s = aggregators; 1–3 min = Twitter/social | vpn07, TradeTheOutcome     |
| **Cross-platform** (Polymarket–Kalshi): 3–5 opportunities per week; 5%+ divergence 15–20% of time; sub-10 ms tools; ~100+ bots competing                                          | PolyTrack, AhaSignals      |
| **Top traders**: Theo4 $22M (88.9% win), Fredi9999 $16.6M, Len9311238 $8.7M (100% win); French trader $85M on 2024 election (proprietary polling)                                 | TradeTheOutcome, Forbes    |
| **Wash trading** fell from 60% (Dec 2024) to ~20% (late 2025); volume-based signals remain noisy                                                                                  | TradeTheOutcome, Columbia  |
| **ICE $2B** investment in Polymarket; $33.4B cumulative volume by Feb 2026                                                                                                        | Coira, Forbes              |

**How we compare:**

| 2026 leading practice                                                 | Us                                                                        | Note                                                                                                               |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Information arb via **primary-source speed** (gov, court, regulatory) | Tape = signal; news too slow [50](50_virix_polymarket_strategy_review.md) | Different wedge. They beat the market to news; we filter on trade flow. We do not build primary-source monitoring. |
| **Cross-platform** (Polymarket–Kalshi) arb                            | Not in scope                                                              | Single-platform focus. Could add later; would require Kalshi API and sub-10 ms tooling. Conscious bypass for now.  |
| **Domain expertise** (politics, sports, crypto) + systematic sizing   | Long Tail target; RecGPT + Gatekeeper                                     | Match.                                                                                                             |
| **Proprietary polling** (e.g. French trader $85M)                     | Not in scope                                                              | Bespoke data; we filter from Tape, not predict from polls.                                                         |
| **Wash trading** inflates reported volume                             | Tape from trade legs                                                      | Consider filtering or downweighting low-confidence volume if the Gatekeeper sees spurious patterns.                |

---

## 3. Inference-scaling: what it is and whether we want it

**What is inference-scaling?** Toby Ord defines it as using _more compute per query_—for example, letting a model "think" longer (more tokens) before answering. The benefit comes from more time, not necessarily more intelligence. The cost is ongoing: you pay every time you run it, and you cannot amortize it over training.

**For reasoning models (e.g. o1, o3):** Most gains come from inference-scaling. Deployment cost multiplies (e.g. 30× more tokens per answer).

### 3.1 Do we inference-scale?

| Component           | Inference-scaling? | Why                                                                                                                                                                         |
| ------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RecGPT**          | No                 | Fixed output size; fixed steps. One recommendation = fixed compute.                                                                                                         |
| **Qwen Gatekeeper** | **Candidate**      | Currently limited to 64 tokens. We could increase to 256, 512, or more and let it reason longer over the Tape before approving or vetoing. That would be inference-scaling. |
| **Beam width**      | Marginal           | More beam = more candidates; slightly more compute. We are latency-capped.                                                                                                  |

The **Gatekeeper is the only place** we can inference-scale: give it more tokens to consider the Tape (trade legs, order book) before deciding.

### 3.2 Should we inference-scale the Gatekeeper?

**Maybe.** Trade-offs:

| For                                                                                                                               | Against                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| More reasoning over a dense Tape could improve trap vs organic discrimination                                                     | The output is binary; it is unclear whether extra thinking helps a yes/no.                                      |
| We have 2–10 s or 10 min windows; we are not latency-bound to 64 tokens                                                           | Cost: more tokens = more cost per Gatekeeper call; multiplies deployment cost.                                  |
| Ord: inference-scaling helps via "more time." The Gatekeeper could use that time to parse gamed-ness, liquidity, leader patterns. | We are reward-data-limited (~100k scenarios). Longer reasoning may not generalize without more training signal. |
| The Tape can be dense (many legs, spreads, agents); 64 tokens may underuse it.                                                    | Start simple; validate at 64 tokens first; A/B test longer reasoning if needed.                                 |

**Verdict:** The Gatekeeper is the only component where inference-scaling makes sense. Do not do it by default. Validate at 64 tokens first. If discrimination plateaus or the Tape is underused, experiment with 256–512 tokens and measure Trap Escape / Organic Strike vs. cost. Keep RecGPT fixed.

---

## 4. Strategic recommendations

### 4.1 Inference-scaling: Gatekeeper only

- Start the Gatekeeper at 64 tokens; validate first.
- If discrimination plateaus, experiment with 256–512 tokens; measure Trap Escape / Organic Strike vs. cost.
- Keep RecGPT fixed; no inference-scaling there.

### 4.2 Double down on our wedge

| Action                      | Rationale                                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Build implication graph** | Leading firms use heuristic reduction. We have RecGPT sequences and a catalog. Build `mix recgpt.build_implication_graph` with sequence-order edges; filter by timeliness. |
| **Extend catalog**          | Add `condition_id` to the catalog. Blocking dependency for butterfly and implication graph.                                                                                |
| **Profit-based Gatekeeper** | IsGamed may be unobservable. Design for profit-only reward from day one. [73](73_gatekeeper_data_scale_and_clob.md)                                                        |
| **Paper trade first**       | Rope bridge: prove survivorship before going live. Validate edge filter and Kelly before putting capital at risk.                                                          |

### 4.3 Avoid the speed race

- Do not chase sub-100 ms. For Binary/Bundle, bypass RecGPT.
- For Catalyst/Combinatorial, use RecGPT + Gatekeeper. Our 250 ms uses ~5–25% of a 2–10 s window. Plenty of room.

### 4.4 Invest in signal, not compute

- **Signal:** Implication graph, catalog, Tape, resolved outcomes. Ord: RL is data-limited; we are too. More resolved markets mean more Gatekeeper scenarios. A better catalog means better butterfly legs.
- **Compute:** RecGPT stays fixed. The Gatekeeper is the only inference-scaling lever (more tokens); use it only if the 64-token baseline plateaus.

---

## 5. Summary

| Dimension             | Decision                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Inference-scaling** | Gatekeeper only; validate at 64 tokens first; experiment with 256–512 if needed. RecGPT fixed.                     |
| **Strategy**          | Mid-Tail + Long Tail; bypass Fat Head.                                                                             |
| **Edge**              | Sequence-based recommendation + learned filter + implication graph; opportunity windows where speed does not bind. |
| **Priority**          | Catalog (`condition_id`), implication graph, profit-based Gatekeeper, paper trade.                                 |

---

## See also

- [50 Virix Polymarket strategy](50_virix_polymarket_strategy_review.md)
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md)
- [68 Wallet longitudinal](68_wallet_longitudinal_adversary_research.md), [69 Sniper longitudinal leaders](69_sniper_longitudinal_leaders.md)
- [70 RL scaling](70_rl_scaling_constraining_diff.md)
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md)
- [73 Gatekeeper data scale](73_gatekeeper_data_scale_and_clob.md)
- [Evidence that Recent AI Gains are Mostly from Inference-Scaling](https://www.tobyord.com/writing/mostly-inference-scaling) — Toby Ord
