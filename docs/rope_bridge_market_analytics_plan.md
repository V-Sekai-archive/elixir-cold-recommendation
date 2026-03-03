# Rope Bridge: Market Analytics and Paper Trading Plan

Paper trade without going bankrupt so we advance to busy-road and multilane-highway stages. This doc covers survivorship (Kelly, Greeks, shock tests), profit calculation from Scout data, Scout-to-butterfly flow, solver options, and the path to profitability.

Related: [thirdparty_bs_p_review](thirdparty_bs_p_review.md), [strategy_given_latency_ceiling](strategy_given_latency_ceiling.md), [44_integer_programming_arbitrage_reference](../thirdparty/reflex-logic-market/polymarket/docs/44_integer_programming_arbitrage_reference.md).

---

## 1. Problem Statement

The rope-bridge paper-trading system must **successfully paper trade without going bankrupt** so we can progress to the busy-road stage and the multilane-highway stage. Survivorship in paper trading is the gate: if we blow up on paper, we never reach live.

To achieve that, we need survivorship-aware sizing and risk:

- **Kelly** for position sizing (don't overbet)
- **Greeks** for portfolio risk (know our exposure)
- **Shock tests** for stress scenarios (what-if before it happens)

The [bs-p](https://github.com/lubluniky/bs-p) crate (polymarket-kernel, MIT) implements these in C with AVX-512 SIMD. We extract and integrate them into our stack.

Goals:

- Integrate with rope-bridge pipeline (RecGPT top-k, butterfly profit checks, paper log, wallet)
- Enforce survivorship (bankruptcy rule, risk caps) via Kelly, Greeks, shock tests
- Own the implementation (Rustler + extracted code; no external bs-p runtime dependency)

---

## 2. Positive Strategies

- **Survivorship layer**: Kelly sizing, Greeks, shock tests, bankruptcy rule — stay in the market
- **Edge filter**: Only size and place trades where profit checks pass (cost < payoff, implication structure)
- **Wallet + P&amp;L**: Track balance, positions, and paper P&amp;L at resolution
- **Selective trading**: Trade only situations with positive expectancy; skip marginal/low-edge picks
- **Rustler + extracted code**: Own the analytics; no external bs-p runtime dependency

---

## 3. Tombstone Failures (Do Not)

- **Random walk without edge filter**: Trading every top-k RecGPT output regardless of edge — survivorship keeps you alive but negative expectancy still bleeds capital slowly
- **Overbetting**: Ignoring Kelly/sizing — blow up on a few bad outcomes
- **No wallet**: Paper trading without balance or bankruptcy logic — no way to know if you would have survived
- **Picking for profit without survivorship**: Chasing edge without risk controls — one correlated drawdown can erase gains

---

## 4. Profit Calculation: Scout Data to Points Profit

We have Scout data (context_item_ids, top-k item_ids) but need a clear path to **points profit**.

### 4.1 Data Flow

```
Scout (RecGPT)          Catalog                 Price Feed              Profit Calc
-------------           -------                 ----------              -----------
context_item_ids  -->   item_id -->             (market_id,             cost = sum(price_i)
top_k item_ids          (market_id,             outcome_id) --> price    payoff = $1 if we win
                         outcome_id)                                    profit = payoff - cost
```

### 4.2 What We Need

| Layer | Input | Output | Source |
|-------|-------|--------|--------|
| **Catalog** | item_id | (market_id, outcome_id), condition_id | DB or JSON; maps RecGPT items to Polymarket outcomes |
| **Price feed** | (market_id, outcome_id) | bid, ask, or mid (0..1) | Polymarket CLOB API, or mock for paper |
| **Resolution** | (market_id, outcome_id) | 1.0 if won, 0.0 else | Polymarket API or simulated |
| **Assignment** | Valid outcome combo | list of {market_id, outcome_id} we hold | Solver or quick checks |

### 4.3 Formula

From [arb_opt/outcome_model.ex](../thirdparty/reflex-logic-market/arb_opt/lib/arb_opt/outcome_model.ex):

```elixir
cost   = sum(price[key] for key in assignment)
payoff = sum(payoffs[key] for key in assignment)  # payoffs[key] = 1.0 if won, else 0
profit = payoff - cost
```

Per-share: we pay `cost` to buy the legs; at resolution we get $1 per share if our outcome wins. **Points profit** = `profit` (in [0,1] when cost and payoff are in probability units).

### 4.4 Build Order

1. **Catalog**: Ensure item_id maps to (market_id, outcome_id) for Polymarket; extend if needed.
2. **Price feed**: Polymarket CLOB client or stub that returns prices by outcome.
3. **Profit module**: Wrap `ArbOpt.OutcomeModel.profit_at_resolution/3` or equivalent.
4. **Solver**: Finds valid assignments; we compute profit for each and keep only positive.

### 4.5 Butterfly / Multi-Leg

Same formula: `cost` = sum of (price_i × quantity_i) for legs we buy; `payoff` = $1 per share when the resolved outcome is in our payoff set. The solver or explicit construction gives us the assignment.

---

## 5. Scout to Butterfly: Learned Catalogue Intent via Trade Legs

**Problem**: Scout outputs item_ids; we need **pairs/sets of markets** to compute butterfly profit. How does Scout's learned catalogue intent become trade legs?

### 5.1 How RecGPT Encodes Intent

- **Sequential recommendation**: RecGPT predicts next item from context (leader sequences, market text). It learns which items tend to follow each other.
- **Catalog-aware decoder**: Outputs are valid catalog item_ids from a trie.
- **Domain-invariant text**: Same model reasons across geopolitics, macro, digital assets.
- **Implication vs correlation**: RecGPT captures semantic and sequential dependencies, not just numeric correlation.

### 5.2 Scout Output → Pairs/Sets

| Scout gives | We need for butterfly | Mapping |
|-------------|------------------------|---------|
| context_item_ids + top_k item_ids | Pairs/sets of (market_id, outcome_id) | Catalog: item_id → market/outcome |
| Sequence [A, B, C, D] | Same-market outcomes (one event, many buckets) | Catalog groups by condition_id/event |
| Sequence [A, B, C, D] | Cross-market implication (A ⇒ B) | Build dependency graph from catalog |
| Leader-follow sequences | Markets leaders trade together | Use sequence as candidate set |

### 5.3 Build Order: From Scout to Profit

1. **Scout call**: `recommend(context_item_ids, top_k)` → `[item_1, ..., item_k]`
2. **Expand to set**: `S = context ∪ top_k` (or just top_k)
3. **Catalog resolution**: For each item_id in S, resolve to `(market_id, outcome_id, condition_id)`. Group by `condition_id` or implication rules.
4. **Form pairs/sets**:
   - **Same-market butterfly**: One condition with 3+ outcome buckets. Use 3 strikes: lower wing, body, upper wing.
   - **Cross-market implication**: Two conditions (A, B) where A⇒B. Pairs = valid (outcome_A, outcome_B) that aren't excluded.
   - **Rebalancing/bundle**: One market, all outcomes. Check sum(prices) < 1.
5. **Fetch prices**: For each (market_id, outcome_id) in the set, get bid/ask from price feed.
6. **Profit check**: For each pair/set, compute cost and payoff; run solver for valid assignments if needed.
7. **Trade only if profit > 0** (after fees).

### 5.4 References

- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — combinatorial arb; heuristic reduction by timeliness, topical similarity.
- Doc 44: Build graph from **RecGPT dataset** (not full chart).
- [41_butterfly_negrisk_greeks](../thirdparty/reflex-logic-market/polymarket/docs/41_butterfly_negrisk_greeks.md): 4 legs, 3 strikes.

---

## 6. Positive Expectancy: Solver Options

We need to **solve** mixed-integer constraints to determine positive expectancy (cost < payoff, valid outcome combinations).

| Option | How it works | Pros | Cons |
|--------|---------------|------|------|
| **Dantzig (HiGHS)** | Elixir lib wraps HiGHS; LP/MILP in-process. arb_opt already uses it. | Battle-tested, in-process, Hex package | CPU only |
| **MiniZinc** | Declarative .mzn models; compiles to FlatZinc; calls solvers via subprocess. | Flexible modeling, solver-agnostic | External process, I/O overhead |
| **HiGHS in EXLA** | Port simplex to Nx/EXLA; batched LP solve on GPU/CPU. | In-process, GPU batching | Major port; research project |

**Recommendation**: Start with **Dantzig**. Use **MiniZinc** if richer modeling is needed. **EXLA port** only if we hit latency limits.

---

## 7. Market Analytics: Rustler + Extracted bs-p (Primary)

### 7.1 Approach

Use **Rustler** for Elixir–Rust NIF bindings, but **extract** the analytics code from [bs-p](https://github.com/lubluniky/bs-p) into our own crate. No external bs-p runtime dependency.

### 7.2 Structure

```
native/recgpt_analytics/
├── Cargo.toml
├── build.rs
├── src/lib.rs
└── c_src/
    ├── kernel.c, kernel.h
    ├── analytics.c, analytics.h
```

### 7.3 Extraction Steps

1. Copy from [bs-p](https://github.com/lubluniky/bs-p) (or local `thirdparty/bs-p`) into `native/recgpt_analytics/c_src/`: kernel.c/h, analytics.c/h (Kelly, shock, aggregate_greeks).
2. Omit: ring_buffer, order_book_microstructure, implied_belief_volatility unless needed.
3. build.rs: compile C with -O3, optionally -mavx512f.
4. lib.rs: Rustler NIFs that call the C APIs.

### 7.4 Elixir API

`RecGPT.MarketAnalytics`:

- `adaptive_kelly_clip_batch/1`
- `simulate_shock_logit_batch/1`
- `aggregate_portfolio_greeks/1`

### 7.5 Fallback: Pure Nx

If Rustler is undesirable: implement sigmoid, logit, greeks, Kelly, shock in Nx with f64 or fixed-point. Lower throughput but single language.

---

## 8. Path to the Profitable 12.7%

Only ~12.7% of Polymarket users are profitable. How we get there:

| Dimension | What we do |
|-----------|------------|
| **Edge** | RecGPT Scout (text-driven, combinatorial, implication); edge filter (cost < payoff); paper-first validation |
| **Survivorship** | Kelly sizing, Greeks, shock tests, wallet, bankruptcy rule |
| **Execution** | Target catalyst/combinatorial (2–10s windows); avoid simple arb races vs sub-100ms bots |
| **Discipline** | Positive strategies only; tombstone failures; consensus signals over single leaders |
| **Validation** | Paper trade end-to-end; only trade when profit > 0; size with Kelly; enforce bankruptcy rule |

---

## 9. Copytrading Economics (Reality Check)

- ~$40M arb extracted (Apr 2024–Apr 2025); most by sub-100ms bots.
- Combinatorial arb: ~$95K (small but less crowded).
- ~87% of copy traders get rekt; leaders hide from copycats.
- **Our position**: Use leader data as *context* for RecGPT, not blind copytrade. Focus on combinatorial/catalyst where speed is less critical. Paper-trade first; only go live if results justify it.

---

## 10. Summary Table

| Item | Decision |
|------|----------|
| **Problem** | Paper trade without going bankrupt; advance to busy-road and multilane-highway |
| **Primary analytics** | Rustler NIF with extracted bs-p in `native/recgpt_analytics/` |
| **Fallback** | Pure Elixir/Nx port |
| **Profit calc** | cost = sum(prices); payoff = $1 if we win; profit = payoff - cost |
| **Scout → butterfly** | item_ids → catalog → pairs/sets → prices → profit check → trade if profit > 0 |
| **Solver** | Dantzig first; MiniZinc if needed; EXLA port for research |
| **Stack** | Rustler, C (kernel + analytics), Rust (NIF glue), Dantzig for LP/MILP |
| **Path to 12.7%** | Edge filter + survivorship + catalyst focus + discipline |

---

## Links

**External**

- [bs-p (GitHub)](https://github.com/lubluniky/bs-p) — polymarket-kernel: AVX-512 quoting, Kelly, Greeks, shock tests
- [bs-p DOCS.md](https://github.com/lubluniky/bs-p/blob/main/DOCS.md) — math (Kelly, Greeks, shock, logit space)
- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — combinatorial arbitrage on Polymarket

**In-repo (docs/)**

- [thirdparty_bs_p_review](thirdparty_bs_p_review.md) — bs-p tricks we can borrow
- [strategy_given_latency_ceiling](strategy_given_latency_ceiling.md) — RecGPT latency vs constraint framework

**Thirdparty (reflex-logic-market, when present)**

- [44_integer_programming_arbitrage_reference](../thirdparty/reflex-logic-market/polymarket/docs/44_integer_programming_arbitrage_reference.md) — IP/solver pipeline
- [arb_opt/outcome_model.ex](../thirdparty/reflex-logic-market/arb_opt/lib/arb_opt/outcome_model.ex) — profit_at_resolution
- [41_butterfly_negrisk_greeks](../thirdparty/reflex-logic-market/polymarket/docs/41_butterfly_negrisk_greeks.md) — butterfly construction
