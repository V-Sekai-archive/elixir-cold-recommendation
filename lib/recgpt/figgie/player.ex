defmodule RecGPT.Figgie.Player do
  @moduledoc """
  Represents a player in the Figgie game.
  """

  defstruct [
    :name,
    :chips,
    :hand,
    :goal_cards,
    :bonus
  ]

  def new(name, starting_chips \\ 350) do
    %__MODULE__{
      name: name,
      chips: starting_chips,
      hand: [],
      goal_cards: 0,
      bonus: 0
    }
  end
end
