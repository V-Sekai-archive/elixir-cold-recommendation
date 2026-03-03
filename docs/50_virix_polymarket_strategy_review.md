# Review: Virix Labs Polymarket Strategy (Ilya @ilyagordey)

Divergence analysis: Virix vs Sniper Mode. Source: [Ilya | Virix Labs](https://x.com/ilyagordey) thread (Mar 1, 2026).

---

## Divergence

| Virix | Us |
|-------|-----|
| Rule-based scoring (markets, orderbook, wallets, cross, news) | Rules generated from market data; RecGPT Scout + Qwen Gatekeeper (learned) |
| On-chain wallet tracking | Leaders' trade legs; catalog maps to outcomes (surface-level concept substantially better) |
| No explicit veto / trap-avoidance | GRPO on IsGamed; veto on traps |
| Perplexity for news | Market is the truth; Polymarket and NYSE track closer than news. News industry too slow (2006 thinking). |
| No Kelly; narrow focus, small capital | Kelly sizing; [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) survivorship |

---

## Discussion

- **Wallet vs leaders' trade legs**: Similar ways to skin same cat. On-chain vs CLOB tracking differences to explore later.
- **Kelly**: Virix avoids harder engineering (Kelly) by narrowing market focus and keeping capital small; we size with Kelly.
- **News feed**: Market is the truth. Polymarket and NYSE track reality closer than news reports and tickers. Traditional news is too slow—outdated. Use market's smart money (leaders) as the signal.

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
- [40 Pipeline + Continuum](40_sniper_pipeline_continuum.md)
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md)
