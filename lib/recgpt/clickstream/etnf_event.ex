defmodule RecGPT.Clickstream.EtnfEvent do
  @moduledoc """
  ETNF event: (session_id, ord) -> item_id, source_dataset.
  One row per click; natural key (session_id, ord).
  """
  use Ecto.Schema

  @primary_key false
  schema "event" do
    field :session_id, :integer
    field :ord, :integer
    field :item_id, :integer
    field :source_dataset, :string
    timestamps(type: :utc_datetime_usec)
  end
end
