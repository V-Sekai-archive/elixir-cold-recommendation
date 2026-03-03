# Sniper: Path to POLYMARKET_PROFITABLE_PCT

**Constant:** `POLYMARKET_PROFITABLE_PCT` — share of Polymarket users who are profitable (historically ~12.7%; update from data).

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Dimensions

From [60 Rope bridge](60_rope_bridge_market_analytics_plan.md):

| Dimension        | What we do                                                                          |
| ---------------- | ----------------------------------------------------------------------------------- |
| **Edge**         | RecGPT Scout + Gatekeeper veto; edge filter (cost < payoff); butterfly profit check |
| **Survivorship** | Kelly sizing, Greeks, shock tests, wallet, bankruptcy rule                          |
| **Execution**    | Catalyst/Combinatorial (10-min window); avoid Binary/Bundle vs sub-100ms bots       |
| **Discipline**   | Positive expectancy only; veto when gamed-ness exceeds threshold                    |
| **Validation**   | Paper trade first; only go live if results justify                                  |

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — Full plan
- [37 Gamed-ness + Metrics](37_sniper_gamedness_metrics.md) — Veto-adjusted expectancy
