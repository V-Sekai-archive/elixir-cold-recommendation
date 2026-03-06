defmodule RecGPT.Figgie.Bayesian do
  @moduledoc """
  Bayesian inference for Figgie goal suit probabilities.

  Calculates starting priors based on card distributions and hand analysis.
  Used to provide mathematical foundations for FuXi-Linear distillation.
  """

  @suits [:spades, :clubs, :hearts, :diamonds]
  @colors %{spades: :black, clubs: :black, hearts: :red, diamonds: :red}

  @doc """
  Calculates the probability distribution of each suit being the goal suit
  based on the known deck structure and observed hand.

  Returns a map of suit => probability.
  """
  def goal_suit_probabilities(hand) do
    # Total possible deck configurations
    # 12-card suit can be any suit, goal is same color (8 or 10 cards)
    # 4 suits for 12-card, 2 possibilities for goal size
    total_configs = 4 * 2

    # For each possible configuration, calculate likelihood given hand
    probabilities =
      for config <- all_possible_configs() do
        {config, likelihood_given_hand(config, hand)}
      end

    # Normalize to probabilities
    total_likelihood = Enum.sum(Enum.map(probabilities, fn {_, l} -> l end))

    normalized =
      Enum.map(probabilities, fn {config, l} -> {config.goal_suit, l / total_likelihood} end)

    # Sum probabilities for each suit
    Enum.reduce(normalized, %{}, fn {suit, prob}, acc ->
      Map.update(acc, suit, prob, &(&1 + prob))
    end)
  end

  @doc """
  Determines if a hand suggests a particular suit is likely the goal.

  Rule of thumb: Many cards of a suit suggest it's the 12-card common suit,
  making the opposite color suit the goal.
  """
  def analyze_hand_signals(hand) do
    counts = Enum.frequencies(hand)

    # Find suit with highest count (likely common)
    {likely_common, common_count} = Enum.max_by(counts, fn {_, count} -> count end)

    common_color = @colors[likely_common]

    # Goal suit is same color, opposite suit
    opposite_suits =
      @suits |> Enum.filter(fn s -> @colors[s] == common_color and s != likely_common end)

    %{
      likely_common_suit: likely_common,
      common_count: common_count,
      likely_goal_suits: opposite_suits,
      # Based on 10-card hand
      confidence: common_count / 10.0
    }
  end

  @doc """
  Calculates the cards needed to reach the majority bonus threshold.

  Need 6 cards of goal suit for guaranteed pot bonus.
  """
  def majority_threshold(hand, goal_suit) do
    current_count = Enum.count(hand, &(&1 == goal_suit))
    needed = max(0, 6 - current_count)
    %{current: current_count, needed: needed, has_majority: current_count >= 6}
  end

  @doc """
  Generates all possible deck configurations.

  Each config: %{common_suit: suit, goal_suit: suit, goal_size: 8|10}
  """
  def all_possible_configs do
    for common <- @suits,
        goal_size <- [8, 10],
        goal <- @suits |> Enum.filter(fn s -> @colors[s] == @colors[common] and s != common end) do
      %{common_suit: common, goal_suit: goal, goal_size: goal_size}
    end
  end

  @doc """
  Calculates likelihood of a configuration given the observed hand.

  Simplified: assumes uniform prior, likelihood based on card counts.
  """
  def likelihood_given_hand(config, hand) do
    # This is a simplified Bayesian calculation
    # In practice, would use more sophisticated probability modeling
    hand_counts = Enum.frequencies(hand)

    # Likelihood proportional to how well the config explains the hand
    common_count = hand_counts[config.common_suit] || 0
    goal_count = hand_counts[config.goal_suit] || 0

    # Higher likelihood if common suit has more cards
    :math.exp(common_count / 10.0) * :math.exp(goal_count / 10.0)
  end
end
