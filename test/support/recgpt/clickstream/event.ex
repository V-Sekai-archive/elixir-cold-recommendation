defmodule RecGPT.Clickstream.Event do
  @moduledoc "Single click event: session_id, ord (order in session), item_id. UCI Clickstream."
  use Ecto.Schema

  @primary_key {:id, :id, autogenerate: true}
  schema "events" do
    field :session_id, :integer
    field :ord, :integer
    field :item_id, :integer
    timestamps(type: :utc_datetime_usec)
  end
end
