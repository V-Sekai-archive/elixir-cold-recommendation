# Wallet Longitudinal Tracing & Adversary Research

Tracing winning wallets across datasets over time. Automation of pre-1990s grunt work once done by futures and options market first-year employees. Opposition research in its purest form on-chain.

---

## Core Problem

**Can we trace winning wallets longitudinally from within the datasets?**

- Follow a single wallet across time, markets, and venues
- Verify trade sequences to confirm the trading algorithm or code in use
- Infer strategy, execution logic, and risk parameters from observable behavior

---

## Analogy: Clustering Game Inputs

Same problem as **clustering Steam games by input patterns**:

| Domain   | Input                            | Cluster              | Inference                             |
| -------- | -------------------------------- | -------------------- | ------------------------------------- |
| Steam    | CSGO, Rainbow Six, Team Fortress | FPS / tac-shooter    | Common control schema, map awareness  |
| On-chain | Wallet A, B, C                   | Same strategy family | Shared algo, copy-trade, or same firm |

Relating inputs across seemingly disjoint sequences reveals latent structure. A wallet's trade legs form a trajectory; clustering trajectories by shape/time/size yields strategy taxonomy.

---

## Longitudinal Verification

**Hypothesis**: A single wallet's trade history, observed over time, is sufficient to:

1. **Confirm the algorithm** – timing, size curves, market selection match known patterns
2. **Infer the code** – execution style (TWAP, VWAP, limit vs market) encoded in order flow
3. **Classify sophistication** – retail vs quant vs market-maker vs insider

Pre-1990s: Analysts manually tracked order flow and filled out spreadsheets. Now: automated ingestion, sequence modeling, and trajectory clustering.

---

## Adversary Research

**Opposition or adversary research, purest form on-chain.**

- No surveys, no leaks, no interviews
- Only observed behavior: trades, sizes, timing, venues
- Public ledger = reproducible longitudinal verification

Use cases: copy-trade reverse engineering, competitor strategy mapping, leaderboard wallet forensics, trap-avoidance (sniping context).

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
- [50 Virix Polymarket Strategy](50_virix_polymarket_strategy_review.md)
- [60 Rope Bridge Market Analytics](60_rope_bridge_market_analytics_plan.md)
