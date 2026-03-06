defmodule RecGPT.Figgie.Game do
  @moduledoc """
  Represents the state of a Figgie game.
  """

  defstruct [
    :players,
    :deck,
    :suit_counts,
    :pot,
    :trading_phase,
    :goal_suit,
    :trades,
    :bids_offers
  ]
end
