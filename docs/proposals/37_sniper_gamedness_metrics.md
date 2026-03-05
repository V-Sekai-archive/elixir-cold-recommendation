# Sniper: Gamed-ness + Moneyball Metrics

Butterfly arbitrage under continuous gamed-ness; veto-adjusted metrics.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Gamed-ness Is a Spectrum

**All markets are gamed to some degree.** Butterfly arbitrage remains calculable:

```
cost = sum(prices); payoff = $1 if we win; profit = payoff - cost
```

Gamed-ness affects _execution risk_ (slippage, resolution manipulation, liquidity), not the arithmetic. Strategy: compute butterfly profit everywhere, then **size by gamed-ness**.

| Market state | Butterfly profit  | Gamed-ness | Action            |
| ------------ | ----------------- | ---------- | ----------------- |
| profit > 0   | Strong edge       | Low        | Full Kelly        |
| profit > 0   | Edge present      | Medium     | Reduced Kelly     |
| profit > 0   | Edge questionable | High       | Tiny size or skip |
| profit ≤ 0   | No edge           | Any        | Skip              |

---

## Moneyball Metrics

| Metric                       | Definition                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------- |
| **Veto-Adjusted Expectancy** | (Trap Escapes × 2 − Trap Hits × 5 + Organic Wins × 1 − Wrong Vetoes × 1) / N |
| **Trap Escape Rate**         | % of IsGamed=True where action = PICK_0                                      |
| **Organic Strike Rate**      | % of IsGamed=False where action = PICK_ID and is_win                         |
| **Triple-Lock Pass Rate**    | % of trades with Organic Tape + Rule Alignment + Tape as Signal              |
| **Liquidity-Adjusted Veto**  | Veto rate by LiquidityScore bucket                                           |

Optimize for Veto-Adjusted Expectancy and Trap Escape Rate, not raw Hit@1.

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — Profit calc, Kelly
