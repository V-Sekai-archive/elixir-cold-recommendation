defmodule RecGPT.Figgie.Trade do
  @moduledoc """
  Represents a trade action in Figgie.
  """

  defstruct [
    :buyer,
    :seller,
    :suit,
    :price,
    :timestamp
  ]
end