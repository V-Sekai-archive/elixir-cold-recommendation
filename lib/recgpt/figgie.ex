defmodule RecGPT.Figgie do
  @moduledoc """
  Figgie game simulation and arbitrage trading logic.

  Figgie is a card game simulating trading markets. This module provides:
  - Game state management
  - Trading mechanics
  - Arbitrage opportunity detection
  - Integration with RecGPT for predictive trading
  """

  alias RecGPT.Figgie.{Game, Player, Trade}

  @suits [:spades, :clubs, :hearts, :diamonds]
  @colors %{spades: :black, clubs: :black, hearts: :red, diamonds: :red}

  @doc """
  Creates a new Figgie game with the specified number of players.
  """
  def new_game(player_count \\ 4) when player_count in 4..5 do
    deck = generate_deck()
    suit_counts = Enum.frequencies(deck)
    players = for i <- 1..player_count, do: Player.new("Player #{i}", 350)

    %Game{
      players: players,
      deck: deck,
      suit_counts: suit_counts,
      pot: 200,
      trading_phase: false,
      goal_suit: nil,
      trades: []
    }
  end

  @doc """
  Deals cards evenly to all players and starts the trading phase.
  """
  def deal_and_start(game) do
    {dealt_cards, remaining_deck} = deal_cards(game.deck, length(game.players))

    players_with_cards =
      Enum.zip(game.players, dealt_cards)
      |> Enum.map(fn {player, cards} -> %{player | hand: cards} end)

    %Game{game | players: players_with_cards, deck: remaining_deck, trading_phase: true}
  end

  @doc """
  Executes a trade between two players.
  """
  def execute_trade(game, %Trade{} = trade) do
    # Validate trade
    # Update player hands and chip counts
    # Clear all quotes
    # Record trade in history
    # This is a simplified version
    game
  end

  @doc """
  Ends trading and calculates payouts.
  """
  def end_round(game) do
    # Reveal goal suit
    goal_suit = determine_goal_suit(game)

    # Calculate bonuses and pot distribution
    players_with_scores = calculate_scores(game.players, goal_suit)

    # Distribute pot
    {winners, pot_split} = find_winners_and_split(players_with_scores, game.pot)

    updated_players = distribute_payouts(players_with_scores, winners, pot_split, goal_suit)

    %Game{game | players: updated_players, trading_phase: false, goal_suit: goal_suit}
  end

  @doc """
  Detects arbitrage opportunities based on current market state.

  Arbitrage in Figgie includes:
  - Buying undervalued suits likely to be goal
  - Selling overvalued suits
  - Statistical edges based on observed trades
  """
  def detect_arbitrage(game, model_predictions \\ nil) do
    # Analyze current bids/offers
    # Compare with model predictions of goal suit probability
    # Identify mispricings
    # Return recommended trades
    []
  end

  # Private functions

  defp generate_deck do
    # Randomly assign suit sizes: 12, 10, 10, 8 in some order
    suit_sizes = [12, 10, 10, 8] |> Enum.shuffle()
    suit_assignments = Enum.zip(@suits, suit_sizes)

    deck =
      for {suit, size} <- suit_assignments, _ <- 1..size do
        suit
      end

    # Shuffle the deck
    Enum.shuffle(deck)
  end

  defp deal_cards(deck, player_count) do
    # Deal cards as evenly as possible
    cards_per_player = div(length(deck), player_count)
    extra_cards = rem(length(deck), player_count)

    {dealt, remaining} = Enum.split(deck, player_count * cards_per_player + extra_cards)

    dealt_cards =
      for i <- 0..(player_count - 1) do
        start_idx = i * cards_per_player + min(i, extra_cards)
        end_idx = start_idx + cards_per_player + if i < extra_cards, do: 1, else: 0
        Enum.slice(dealt, start_idx, end_idx - start_idx)
      end

    {dealt_cards, remaining}
  end

  defp determine_goal_suit(game) do
    # Count cards per suit
    counts = game.suit_counts
    twelve_card_suit = Enum.find(counts, fn {_suit, count} -> count == 12 end) |> elem(0)
    twelve_color = @colors[twelve_card_suit]

    # Goal suit is same color, either 8 or 10 cards
    same_color_suits = @suits |> Enum.filter(fn s -> @colors[s] == twelve_color end)
    goal_candidates = same_color_suits |> Enum.filter(fn s -> counts[s] in [8, 10] end)
    Enum.random(goal_candidates)
  end

  defp calculate_scores(players, goal_suit) do
    Enum.map(players, fn player ->
      goal_count = Enum.count(player.hand, &(&1 == goal_suit))
      %{player | goal_cards: goal_count, bonus: goal_count * 10}
    end)
  end

  defp find_winners_and_split(players_with_scores, pot) do
    max_goal = Enum.max_by(players_with_scores, & &1.goal_cards).goal_cards
    winners = Enum.filter(players_with_scores, &(&1.goal_cards == max_goal))
    pot_split = div(pot, length(winners))
    {winners, pot_split}
  end

  defp distribute_payouts(players, winners, pot_split, goal_suit) do
    Enum.map(players, fn player ->
      bonus = player.bonus
      pot_share = if Enum.member?(winners, player), do: pot_split, else: 0
      %{player | chips: player.chips + bonus + pot_share}
    end)
  end
end
