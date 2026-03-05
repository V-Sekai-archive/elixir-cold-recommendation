# Rope Bridge: Market Analytics and Paper Trading Plan

Paper trade without going bankrupt so we advance to busy-road and multilane-highway stages. This doc covers survivorship (Kelly, Greeks, shock tests), profit calculation from Scout data, Scout-to-butterfly flow, solver options, and the path to profitability.

Related: [67 Thirdparty bs-p](67_thirdparty_bs_p_review.md), [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md), [44_integer_programming_arbitrage_reference](../thirdparty/reflex-logic-market/polymarket/docs/44_integer_programming_arbitrage_reference.md).

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

| Layer          | Input                   | Output                                  | Source                                               |
| -------------- | ----------------------- | --------------------------------------- | ---------------------------------------------------- |
| **Catalog**    | item_id                 | (market_id, outcome_id), condition_id   | DB or JSON; maps RecGPT items to Polymarket outcomes |
| **Price feed** | (market_id, outcome_id) | bid, ask, or mid (0..1)                 | Polymarket CLOB API, or mock for paper               |
| **Resolution** | (market_id, outcome_id) | 1.0 if won, 0.0 else                    | Polymarket API or simulated                          |
| **Assignment** | Valid outcome combo     | list of {market_id, outcome_id} we hold | Solver or quick checks                               |

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

### 5.0 Can we determine the legs?

We **can** determine legs **iff** we have:

| Requirement                                                     | Status            | Source                                                                                                                          |
| --------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Catalog** `item_id` → `(market_id, outcome_id, condition_id)` | Build required    | Extend items.json or add mapping table; Polymarket datasets (Kaggle, Jon-Becker) have condition_id, outcome structure           |
| **Market structure** (which outcomes share a condition)         | Available in data | Polymarket API, markets.csv, Jon-Becker Parquet: each condition lists its outcomes                                              |
| **Same-market butterfly** (3+ outcomes in one condition)        | Determinable      | Once catalog has condition_id per item, group by condition_id; pick lower/body/upper wing from outcome set                      |
| **Cross-market implication** (A ⇒ B)                            | Build required    | See [§5.4](#54-how-to-build-the-cross-market-implication-graph): RecGPT sequences, topical similarity, timeliness, domain rules |
| **Prices** per `(market_id, outcome_id)`                        | Available         | CLOB API, price feed; needed for profit calc                                                                                    |

**What we know:** Polymarket data contains condition_id, outcome_ids, and market structure. Scout's item_id is our 0-based catalog index. We need a **mapping layer**: item_id → (condition_id, outcome_id). Once that exists, we can group items by condition_id and form same-market butterflies. Cross-market butterflies need an implication graph we have not yet built.

**Measurement:** We can measure butterfly profit **only after** catalog resolution. Until item_id → Polymarket outcome exists, we cannot compute cost or payoff for Scout output.

### 5.1 How RecGPT Encodes Intent

- **Sequential recommendation**: RecGPT predicts next item from context (leader sequences, market text). It learns which items tend to follow each other.
- **Catalog-aware decoder**: Outputs are valid catalog item_ids from a trie.
- **Domain-invariant text**: Same model reasons across geopolitics, macro, digital assets.
- **Implication vs correlation**: RecGPT captures semantic and sequential dependencies, not just numeric correlation.

### 5.2 Scout Output → Pairs/Sets

| Scout gives                       | We need for butterfly                          | Mapping                              |
| --------------------------------- | ---------------------------------------------- | ------------------------------------ |
| context_item_ids + top_k item_ids | Pairs/sets of (market_id, outcome_id)          | Catalog: item_id → market/outcome    |
| Sequence [A, B, C, D]             | Same-market outcomes (one event, many buckets) | Catalog groups by condition_id/event |
| Sequence [A, B, C, D]             | Cross-market implication (A ⇒ B)               | Build dependency graph from catalog  |
| Leader-follow sequences           | Markets leaders trade together                 | Use sequence as candidate set        |

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

### 5.4 Cross-market implication graph: technical design

An edge A ⇒ B means: if condition A resolves true, condition B is more likely (or logically implied) true. Valid pairs for cross-market butterflies are `(outcome_A, outcome_B)` where A⇒B.

#### 5.4.0 How do we build it? Sequence order vs embeddings vs semantic id

| Source                        | Role                | Used for                                                                                                                                                                                                                                |
| ----------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sequence order (item_id)**  | **Primary**         | Train/leader sequences are lists of item_ids. Extract adjacent pairs (a, b): a before b → candidate edge (condition_a, condition_b). Co-occurrence count = weight. No embeddings or semantic id involved.                               |
| **Semantic id (4-token FSQ)** | **Not used**        | RecGPT’s semantic id is the 4-token FSQ code per item. We could cluster items by similar codes and add edges within clusters, but the current design does not. Graph is built from **sequence position**, not from semantic similarity. |
| **Market embeddings (768-d)** | **Optional filter** | Embed market titles with `RecGPT.Embedding`; cosine_sim(A, B) ≥ threshold → keep edge. Use only to **prune** false edges from unrelated markets that co-occur by chance. Not the primary builder.                                       |

**Summary:** We build the graph from **sequence co-occurrence** (item A before item B in train/leader sequences). That gives candidate edges. We optionally **filter** by embeddings (topical similarity) and timeliness (resolution overlap). Semantic id is not used. The graph is **order-based**, not **similarity-based**.

#### 5.4.1 Graph representation

| Structure                  | Format                                                                                                     | Use                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Adjacency list**         | `%{condition_id_a => [condition_id_b, ...]}`                                                               | Forward edges; O(1) successor lookup                           |
| **Edge list with weights** | `[{c_a, c_b, weight}, ...]`                                                                                | Sorting, filtering; weight = co-occurrence count or cosine sim |
| **JSON output**            | `{"edges": [{"from": "cid_a", "to": "cid_b", "weight": n, "source": "sequence"}], "condition_ids": [...]}` | Persistence; downstream consumers                              |

Vertices = `condition_id` (Polymarket condition). Catalog maps `item_id` → `condition_id`; multiple item_ids may map to same condition (e.g. different outcome buckets).

#### 5.4.2 Algorithms

**Transition extraction (from train_sequences.json):**

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

**Optional: k-step transitions** — for sequence `[A, B, C, D]`, add (A,D) as well as (A,B), (B,C), (C,D) to capture longer-range implication. Weight by `1 / (k+1)` for decay.

**Topical filter (optional):** Embed market titles with `RecGPT.Embedding` or Bumblebee; cosine_sim(market_A, market_B) ≥ threshold → keep edge. Reduces false edges from unrelated markets that happen to co-occur.

**Timeliness filter:** Join with Polymarket market metadata; keep only edges where `resolution_end_a` and `resolution_end_b` overlap within a window (e.g. ±7 days). Prevents pairing markets that resolve in different epochs.

#### 5.4.3 Proposed tooling

| Tool                                 | Purpose                                | Input                                          | Output                                   |
| ------------------------------------ | -------------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| `mix recgpt.build_implication_graph` | Extract edges from sequences + catalog | `--train`, `--catalog`, `--min-count`, `--out` | `implication_graph.json`                 |
| `RecGPT.ImplicationGraph` module     | Build, merge, filter graph             | Same                                           | In-memory graph or JSON                  |
| Catalog schema extension             | Store condition_id per item            | `items.json` or `item_condition_mapping.json`  | `item_id` → `(condition_id, outcome_id)` |

**Mix task contract:**

```
mix recgpt.build_implication_graph \
  --train data/polymarket/train_sequences.json \
  --catalog data/polymarket/items.json \
  --min-count 5 \
  --out data/polymarket/implication_graph.json
```

**Output schema (implication_graph.json):**

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

#### 5.4.4 Catalog requirement

Catalog must include `condition_id` (and `outcome_id`) per item. Extend `items.json`:

```json
{ "id": 0, "title": "...", "condition_id": "0x...", "outcome_id": "0x..." }
```

Or separate mapping file `item_condition_mapping.json`:

```json
{
  "mappings": [{ "item_id": 0, "condition_id": "0x...", "outcome_id": "0x..." }]
}
```

#### 5.4.5 Build pipeline

1. **Catalog** — Ensure item_id → condition_id exists (from Polymarket fetch or manual mapping).
2. **Extract** — `mix recgpt.build_implication_graph` reads train_sequences, emits candidate edges.
3. **Merge** (optional) — Append domain rules from `domain_rules.json` (hand-curated edges).
4. **Filter** — Apply timeliness (resolution overlap) and topical similarity if metadata/embeddings available.
5. **Validate** — Backtest: for each edge, check if (outcome_A, outcome_B) ever yielded profit in historical resolution. Drop edges that never do.

#### 5.4.6 Current status

No implementation exists. `RecGPT.Catalog.Sync` and `RecGPT.PretrainRunner` load train_sequences but do not extract transitions. Catalog schema has `item_id` and `title` only; no `condition_id`. Build order: extend catalog schema → implement `RecGPT.ImplicationGraph.extract_from_sequences/2` → add mix task → add filters.

**Reference:** [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — heuristic reduction by timeliness, topical similarity, combinatorial relationships; validated by expert input.

### 5.5 References

- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — combinatorial arb; heuristic reduction by timeliness, topical similarity.
- [41_butterfly_negrisk_greeks](../thirdparty/reflex-logic-market/polymarket/docs/41_butterfly_negrisk_greeks.md): 4 legs, 3 strikes.
- [76 Generalized pairing sources](76_generalized_pairing_sources.md) — arXiv gold standard, dhruv575 CSV, build-your-own pairing pipeline.

---

## 6. Positive Expectancy: Solver Options

We need to **solve** mixed-integer constraints to determine positive expectancy (cost < payoff, valid outcome combinations).

| Option              | How it works                                                                 | Pros                                   | Cons                           |
| ------------------- | ---------------------------------------------------------------------------- | -------------------------------------- | ------------------------------ |
| **Dantzig (HiGHS)** | Elixir lib wraps HiGHS; LP/MILP in-process. arb_opt already uses it.         | Battle-tested, in-process, Hex package | CPU only                       |
| **MiniZinc**        | Declarative .mzn models; compiles to FlatZinc; calls solvers via subprocess. | Flexible modeling, solver-agnostic     | External process, I/O overhead |
| **HiGHS in EXLA**   | Port simplex to Nx/EXLA; batched LP solve on GPU/CPU.                        | In-process, GPU batching               | Major port; research project   |

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

| Dimension        | What we do                                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| **Edge**         | RecGPT Scout (text-driven, combinatorial, implication); edge filter (cost < payoff); paper-first validation |
| **Survivorship** | Kelly sizing, Greeks, shock tests, wallet, bankruptcy rule                                                  |
| **Execution**    | Target catalyst/combinatorial (2–10s windows); avoid simple arb races vs sub-100ms bots                     |
| **Discipline**   | Positive strategies only; tombstone failures; consensus signals over single leaders                         |
| **Validation**   | Paper trade end-to-end; only trade when profit > 0; size with Kelly; enforce bankruptcy rule                |

---

## 9. Copytrading Economics (Reality Check)

- ~$40M arb extracted (Apr 2024–Apr 2025); most by sub-100ms bots.
- Combinatorial arb: ~$95K (small but less crowded).
- ~87% of copy traders get rekt; leaders hide from copycats.
- **Our position**: Use leader data as _context_ for RecGPT, not blind copytrade. Focus on combinatorial/catalyst where speed is less critical. Paper-trade first; only go live if results justify it.

---

## 10. Summary Table

| Item                  | Decision                                                                       |
| --------------------- | ------------------------------------------------------------------------------ |
| **Problem**           | Paper trade without going bankrupt; advance to busy-road and multilane-highway |
| **Primary analytics** | Rustler NIF with extracted bs-p in `native/recgpt_analytics/`                  |
| **Fallback**          | Pure Elixir/Nx port                                                            |
| **Profit calc**       | cost = sum(prices); payoff = $1 if we win; profit = payoff - cost              |
| **Scout → butterfly** | item_ids → catalog → pairs/sets → prices → profit check → trade if profit > 0  |
| **Solver**            | Dantzig first; MiniZinc if needed; EXLA port for research                      |
| **Stack**             | Rustler, C (kernel + analytics), Rust (NIF glue), Dantzig for LP/MILP          |
| **Path to 12.7%**     | Edge filter + survivorship + catalyst focus + discipline                       |

---

## Links

**External**

- [bs-p (GitHub)](https://github.com/lubluniky/bs-p) — polymarket-kernel: AVX-512 quoting, Kelly, Greeks, shock tests
- [bs-p DOCS.md](https://github.com/lubluniky/bs-p/blob/main/DOCS.md) — math (Kelly, Greeks, shock, logit space)
- [Unravelling the Probabilistic Forest](https://arxiv.org/abs/2508.03474) — combinatorial arbitrage on Polymarket

**In-repo (docs/)**

- [67 Thirdparty bs-p](67_thirdparty_bs_p_review.md) — bs-p tricks we can borrow
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) — RecGPT latency vs constraint framework

**Thirdparty (reflex-logic-market, when present)**

- [44_integer_programming_arbitrage_reference](../thirdparty/reflex-logic-market/polymarket/docs/44_integer_programming_arbitrage_reference.md) — IP/solver pipeline
- [arb_opt/outcome_model.ex](../thirdparty/reflex-logic-market/arb_opt/lib/arb_opt/outcome_model.ex) — profit_at_resolution
- [41_butterfly_negrisk_greeks](../thirdparty/reflex-logic-market/polymarket/docs/41_butterfly_negrisk_greeks.md) — butterfly construction
