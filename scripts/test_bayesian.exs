#!/usr/bin/env elixir

# Test script for Figgie Bayesian calculations

# Sample hand: 6 hearts, 2 diamonds, 1 spade, 1 club
hand = [:hearts, :hearts, :hearts, :hearts, :hearts, :hearts, :diamonds, :diamonds, :spades, :clubs]

IO.puts("Sample hand: #{inspect(hand)}")
IO.puts("Hand analysis:")
analysis = RecGPT.Figgie.Bayesian.analyze_hand_signals(hand)
IO.inspect(analysis)

IO.puts("\nGoal suit probabilities:")
probabilities = RecGPT.Figgie.Bayesian.goal_suit_probabilities(hand)
IO.inspect(probabilities)

IO.puts("\nMajority threshold for diamonds (assuming it's goal):")
threshold = RecGPT.Figgie.Bayesian.majority_threshold(hand, :diamonds)
IO.inspect(threshold)