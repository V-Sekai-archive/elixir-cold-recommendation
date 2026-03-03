# Sniper Mode + Longitudinal Leader Tracing

Applying [68 Wallet Longitudinal Adversary Research](68_wallet_longitudinal_adversary_research.md) to [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md): use on-chain wallet trajectory tracing to sharpen leaders, trap-avoidance, and the Scout→Gatekeeper pipeline.

---

## Mapping

| 68 Concept | Sniper Application |
|------------|---------------------|
| Trace winning wallets longitudinally | **Leaders' trade legs** – follow Fat Head / Mid-Tail leaders across time, markets, venues |
| Cluster trajectories by shape/time/size | **Pipeline N+2 (cluster)** – cluster leaders by execution style, not just catalog items |
| Infer algorithm from observable behavior | **Scout catalog** – trajectory taxonomy maps to outcomes; RecGPT learns which trajectories win |
| Adversary research, trap-avoidance | **Gatekeeper veto** – longitudinal trace of trap-hitters vs trap-escapers → calibrate GRPO reward |
| Confirm algo via timing/size curves | **Butterfly arbitrage** – leaders who size consistently with known patterns get higher confidence |

---

## Leaders as Longitudinal Objects

**Leaders** (polymarket top performers) are wallets we can trace:

1. **Across time** – same wallet, many resolutions; win rate, drawdown, recovery
2. **Across markets** – which markets they enter, which they avoid (gamed-ness signal)
3. **Across venues** – Polymarket, prediction markets, CEX if linked

Pre-1990s grunt work: hand-tracking order flow. Now: leaders' trade legs become sequences; RecGPT pretrains on them; trajectory clustering yields strategy families.

---

## Trap-Avoidance from Longitudinal Verification

**Hypothesis**: Wallets that repeatedly escape traps have distinguishable trajectory patterns from those that hit them.

- **Trap-hitters** – timing/size/market selection converges to predictable "sucker" clusters
- **Trap-escapers** – veto-like behavior encoded in order flow (hesitation, size-down, skip)

Adversary research: trace both cohorts longitudinally. Feed escape patterns into Qwen Gatekeeper GRPO (asymmetric reward: -5 trap hit, +2 trap veto). Longitudinal verification = reproducible proof that a leader's algo is trap-aware.

---

## Pipeline Integration

| Step | Before | After (68 applied) |
|------|--------|--------------------|
| N (catalog) | Items, markets | + Leader IDs, wallet→outcome mappings |
| N+1 (RecGPT) | Pretrain on sequences | + Trajectory-conditioned sequences (timing, size curves) |
| N+2 (cluster) | Cluster by item co-occurrence | + Cluster by **trajectory shape** (same strategy family = same cluster) |
| N+3 (Qwen LoRA) | GRPO on IsGamed, veto | + GRPO reward informed by longitudinal trap-hit/escape rates |

---

## Strategy Taxonomy from Trajectories

Like clustering CSGO / Rainbow Six / Team Fortress by input schema:

| Trajectory Cluster | Inference | Sniper Use |
|--------------------|-----------|------------|
| Fat Head, consistent size, early entry | Market-maker or insider | Bypass Scout; high confidence |
| Mid-Tail, variable size, veto patterns | Quant / trap-aware | Scout + Gatekeeper; normal flow |
| Long Tail, random size, late entry | Retail / copy-trade | Early edge or skip |
| Trap-hitter profile | Predictable sucker pattern | Gatekeeper veto; negative signal |

---

## See Also

- [68 Wallet Longitudinal Adversary Research](68_wallet_longitudinal_adversary_research.md)
- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
- [50 Virix Polymarket Strategy](50_virix_polymarket_strategy_review.md) — wallet vs leaders' trade legs
- [37 Gamed-ness + Metrics](37_sniper_gamedness_metrics.md)
- [39 Triple-Lock + Execution](39_sniper_triple_lock_execution.md)
