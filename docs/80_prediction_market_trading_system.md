# Prediction market trading system

When writing improvement proposals, follow this structure. Apply rope bridge then busy road then highway - incremental scaling.

## Domain

Prediction market trading. **Strategy: Scout-first** - RecGPT Scout is the core; we use leader data as *context* for RecGPT, not blind copytrade. Flow: leader sequences (context) -> Scout inference -> top-k item_ids -> catalog lookup -> Polymarket outcomes -> price feed (CLOB) -> profit calc (payoff - cost). We want a rope bridge: paper trade without going bankrupt before advancing to live trading. **Scout must be trained on leader/prediction-market data** - an off-domain RecGPT has zero signal for Polymarket outcomes; rope bridge includes minimal Scout training to get signal.

## Architecture

The stack uses Elixir, Ecto, reflex-logic-market (arb_opt, outcome_model). Data flows: Scout context_item_ids and top-k item_ids -> catalog (item_id to market_id, outcome_id) -> price feed (CLOB or mock) -> profit calculation (cost, payoff, profit).

## Alignment

**Target: Rope bridge first.** Paper trading must survive (no bankruptcy) before busy-road (live with small capital) or multilane-highway (scaled trading, SPMD). Related: [60 Rope bridge market analytics](60_rope_bridge_market_analytics_plan.md), [77 Rope bridge analogy](77_rope_bridge_analogy_zguide.md).

## Problem or limitation

We need to prove the prediction market trading path works. There are thousands of markets; even finding and selecting one market type is a rope bridge problem. Scout gives us picks; we need a minimal end-to-end flow from market selection to picks to paper P&amp;L. Without survivorship (Kelly, Greeks, shock tests, bankruptcy rule), we cannot know if the strategy would survive before going live. Without a clear profit calculation (cost, payoff from resolution), we cannot evaluate edge. Workarounds: none yet. Related: [60 Rope bridge](60_rope_bridge_market_analytics_plan.md), [68 Wallet longitudinal](68_wallet_longitudinal_adversary_research.md), [69 Sniper longitudinal leaders](69_sniper_longitudinal_leaders.md), [73 Gatekeeper CLOB](73_gatekeeper_data_scale_and_clob.md).

## Proposed improvement

Build prediction market trading in stages, aligned to rope bridge discipline.

**Rope bridge progression (prediction markets):**

| Stage             | Meaning                                                                                                                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rope bridge       | Minimal end-to-end path: **Pick one market** (from thousands). **Identify leaders** and use their sequences as context (longitudinal wallet tracing or leaderboard). **Train Scout** on leader sequences (minimal training for signal; off-domain RecGPT has none). Then: leader context -> Scout inference -> top-k -> catalog lookup -> one price source (mock or CLOB) -> profit calc (cost, payoff). Add survivorship: Kelly sizing, bankruptcy rule, paper wallet. One market, one leader cohort. Goal: paper trade Scout output without going bankrupt. |
| Busy road         | Live trading with small capital. Real CLOB, real execution. More markets, pipelining. Expand Scout training (more leaders, more markets). Harden with Greeks and shock tests.                                                                                              |
| Multilane highway | Scaled trading: SPMD, sharding, high throughput. Multi-rank eval, distributed serving.                                                                                                                                                                                   |

The sequence is strict: rope bridge first. If the rope bridge fails (bankruptcy on paper), you never reach the road or highway.

**Minimal rope bridge (prediction markets):**

- **Select one market** from thousands - filter/criteria to choose is part of the rope bridge; don't assume it's given
- **Identify leaders** and use their sequences as context - which wallets/leaders; leader selection is part of the rope bridge
- **Scout training** - train on leader sequences for the one market/cohort (off-domain RecGPT has 0 signal)
- **Scout inference** - leader context -> trained Scout -> top-k item_ids
- One outcome assignment path (solver or quick checks)
- Catalog: item_id -> (market_id, outcome_id)
- Price feed: mock or single CLOB endpoint
- Profit: cost = sum(price), payoff = sum(resolved), profit = payoff - cost
- Survivorship: Kelly sizing, bankruptcy rule, paper wallet with balance

## Proposal review

Ask yourself honestly: Is your description above specific enough for contributors to implement this improvement without improvising the details?

If not, consider opening a discussion first instead. Repeatedly posting unactionable or low-effort proposals may lead to restrictions on your ability to post.

## Glossary

| Term              | Definition                                                                                                                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scout**         | RecGPT output: context_item_ids and top-k item_ids. Must be trained on leader/prediction-market data to have signal; off-domain RecGPT has none. Uses leader sequences as context at inference. The picks we use for prediction market positions.                                                                      |
| **Copytrade**     | Following the trades of leaders (top performers). We use leader data as context for Scout, not blind copytrade; Scout generates picks.                                                                                                                           |
| **Leaders**       | Top-performing wallets or traders. We identify them via longitudinal tracing, leaderboards, or performance metrics; their sequences become context for Scout inference.                                                                                            |
| **CLOB**          | Central Limit Order Book. The order book (bids, asks) for a market. Polymarket uses a CLOB for trading.                                                           |
| **Polymarket**    | A prediction market platform. We trade binary outcome tokens (prices 0..1).                                                                                      |
| **Kelly**         | Kelly criterion for position sizing. Prevents overbetting; used for survivorship.                                                                                |
| **Greeks**        | Portfolio risk metrics (exposure, sensitivity). Used for stress testing.                                                                                          |
| **Shock tests**   | What-if scenarios. Stress the portfolio before it happens.                                                                                                        |
| **Payoff**        | Payout at resolution. For a winning outcome, payoff = 1.0; else 0.0.                                                                                               |
| **Catalog**       | Mapping from item_id (RecGPT) to (market_id, outcome_id) on Polymarket.                                                                                          |
| **Outcome model** | In arb_opt: assigns valid outcome combinations for a market; used for profit calculation.                                                                        |
| **Paper trading** | Trading with simulated money. No real capital at risk; used to validate strategy before live.                                                                    |
| **SPMD**          | Single Program Multiple Data. Parallel execution; used at multilane-highway stage.                                                                               |
