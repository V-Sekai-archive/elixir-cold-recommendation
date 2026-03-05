# 95 Figgie Arbitrage Trading

## Overview

RecGPT has been extended to support **arbitrage trading** in Figgie, a real-time card game that simulates market dynamics. The system uses sequential recommendation techniques to model trade sequences and predict profitable arbitrage opportunities.

## Figgie Game Mechanics

Figgie is played with 4-5 players using a 40-card deck:
- Suits: Spades ♠ (black), Clubs ♣ (black), Hearts ♥ (red), Diamonds ♦ (red)
- Distribution: 12-card suit, 10-card suit, 10-card suit, 8-card suit
- Goal suit: Same color as 12-card suit, worth 8 or 10 cards
- Trading: 4-minute rounds with real-time bidding/offering
- Payout: $10/card bonus + pot split for most goal cards

## Arbitrage Opportunities

The system identifies arbitrage through:

1. **Goal Suit Prediction**: Model infers 12-card suit from trade patterns
2. **Expected Value Calculation**: Computes fair prices based on goal probabilities
3. **Mispricing Detection**: Identifies suits trading above/below fair value
4. **Statistical Edges**: Exploits incomplete information in real-time trading

## Implementation

### Core Modules

- `RecGPT.Figgie` - Game simulation and state management
- `RecGPT.Figgie.Trading` - Arbitrage detection and trade execution
- `RecGPT.Figgie.DataFetcher` - Training data generation

### Data Format

Figgie sequences are formatted as `[buyer_id, seller_id, suit_index, price]`:

```json
{
  "items": ["spades", "clubs", "hearts", "diamonds"],
  "sequences": [
    [[2, 0, 1, 94], [2, 0, 1, 22], [0, 1, 0, 52]]
  ]
}
```

### Training Pipeline

```bash
# Generate simulated game data
mix recgpt.figgie_simulate --games 10000 --output priv/figgie_fixture.json

# Train the model
mix recgpt.pretrain --fixture priv/figgie_fixture.json --output figgie_checkpoint

# Evaluate performance
mix recgpt.eval --fixture priv/figgie_fixture.json --checkpoint figgie_checkpoint
```

## Performance Optimizations

For real-time arbitrage (target: 20ms P50), the system implements:

- **SId Caching**: Pre-compute suit embeddings in ETS
- **Static Graphs**: EXLA JIT with fixed shapes for RNN-style updates
- **Kernel Fusion**: Merge FSQ projection into FuXi-Linear layers
- **GPU Acceleration**: CUDA backend for sub-millisecond inference

## Trading Strategies

### Basic Strategy
- Infer 12-card suit from observed card counts
- Accumulate likely goal suits at discount
- Sell non-goal suits at premium

### Advanced Arbitrage
- Model-based prediction of goal suit probabilities
- Statistical arbitrage across correlated suits
- Market making between bid/ask spreads

## Integration Points

### Live Trading
```elixir
# Load trained model
{:ok, model} = RecGPT.Inference.load_checkpoint("figgie_checkpoint")

# Analyze current game state
opportunities = RecGPT.Figgie.Trading.find_arbitrage_opportunities(game, model)

# Execute profitable trades
Enum.each(opportunities, &RecGPT.Figgie.Trading.execute_arbitrage_trade(game, &1))
```

### Simulation Testing
```elixir
# Run automated tournaments
results = RecGPT.Figgie.Simulation.run_tournament(bot_count: 4, games: 100)
IO.inspect(results, label: "Arbitrage Performance")
```

## Metrics and Evaluation

- **Profit Rate**: Average profit per trade vs. random baseline
- **Win Rate**: Percentage of games with positive returns
- **Latency**: P50/P99 inference times for real-time trading
- **Arbitrage Efficiency**: Percentage of identified opportunities executed profitably

## Future Extensions

- **Multi-Game Arbitrage**: Simultaneous play across multiple Figgie instances
- **Human-AI Hybrid**: Assist human players with real-time recommendations
- **Market Making**: Provide liquidity through automated bid/ask management
- **Portfolio Optimization**: Multi-suit position management across rounds

## References

- [Figgie Rules](https://www.figgie.com/how-to-play.html)
- [Performance Optimizations](figgie_performance_optimization.md)
- [RecGPT Paradigm](11_recgpt_paradigm.md)