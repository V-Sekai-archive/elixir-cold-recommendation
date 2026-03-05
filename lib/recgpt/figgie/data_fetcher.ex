defmodule RecGPT.Figgie.DataFetcher do
  @moduledoc """
  Fetches or generates Figgie game data for training RecGPT.

  Since Figgie is a game without public APIs, this module:
  - Simulates complete games to generate training sequences
  - Converts game states into RecGPT-compatible sequences
  - Provides arbitrage-labeled data for supervised learning
  """

  alias RecGPT.Figgie

  @doc """
  Generates training data by simulating Figgie games.

  Returns sequences of trades with arbitrage opportunities labeled.
  """
  def fetch_training_data(game_count \\ 1000) do
    for _ <- 1..game_count do
      simulate_game_with_arbitrage_labels()
    end
  end

  @doc """
  Simulates a complete Figgie game and labels arbitrage opportunities.
  """
  def simulate_game_with_arbitrage_labels do
    game = RecGPT.Figgie.new_game() |> RecGPT.Figgie.deal_and_start()

    # Simulate trading phase with random trades
    game_after_trading = simulate_trading_phase(game)

    # End round and get final state
    final_game = RecGPT.Figgie.end_round(game_after_trading)

    # Extract sequence and labels
    build_labeled_sequence(final_game)
  end

  @doc """
  Converts Figgie game data into RecGPT fixture format.
  """
  def to_fixture_data(games_data) do
    # Convert to the format expected by RecGPT training
    # Items = suits/contracts, sequences = trade histories
    sequences = Enum.map(games_data, fn {sequence, _labels} -> sequence end)

    %{
      items: ["spades", "clubs", "hearts", "diamonds"],
      sequences: sequences
    }
  end

  @doc """
  Converts labeled Figgie game data into enhanced fixture format with pattern labels.
  """
  def to_labeled_fixture_data(games_data) do
    sequences = Enum.map(games_data, fn {sequence, labels} -> sequence end)
    pattern_labels = Enum.map(games_data, fn {_sequence, labels} -> labels end)

    %{
      items: ["spades", "clubs", "hearts", "diamonds"],
      sequences: sequences,
      pattern_labels: pattern_labels
    }
  end

  @doc """
  Generates training data by simulating Figgie games with comprehensive pattern labels.
  """
  def fetch_labeled_training_data(game_count \\ 1000) do
    for _ <- 1..game_count do
      simulate_game_with_pattern_labels()
    end
  end

  @doc """
  Simulates a complete Figgie game and labels trading patterns and arbitrage opportunities.
  """
  def simulate_game_with_pattern_labels do
    # TODO: Implement pattern-based game simulation
    {:error, :not_implemented}
  end

  # Private functions

  defp simulate_trading_phase(game, _time_remaining \\ 240) do
    # TODO: Implement strategic trading phase
    game
    # Simulate random trades for the duration
    # Generate trades with incremental timestamps
    start_time = DateTime.utc_now()
    trades = for i <- 1..10 do
      %RecGPT.Figgie.Trade{
        buyer: Enum.random(0..3),
        seller: Enum.random(0..3),
        suit: Enum.random([:spades, :clubs, :hearts, :diamonds]),
        price: Enum.random(10..100),
        timestamp: DateTime.add(start_time, i * 1000, :millisecond)  # 1 second apart
      }
    end

    %{game | trades: trades}
  end

  defp build_labeled_sequence(game) do
    # Build sequence of trades with time_ms
    sequence = Enum.map(game.trades, fn trade ->
      suit_index = case trade.suit do
        :spades -> 0
        :clubs -> 1
        :hearts -> 2
        :diamonds -> 3
      end
      time_ms = DateTime.to_unix(trade.timestamp, :millisecond)
      [suit_index, time_ms]
    end)

    # Label arbitrage opportunities (simplified - mark all trades as potential arbitrage)
    labels = Enum.map(sequence, fn _ -> 1 end)

    {sequence, labels}
  end
end