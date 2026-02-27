defmodule RecGPT.Clickstream.Item do
  @moduledoc "Catalog item for UCI Clickstream (e-shop clothing). id 0-based, title = item text for RecGPT."
  use Ecto.Schema

  @primary_key {:id, :integer, autogenerate: false}
  schema "items" do
    field :title, :string
    timestamps(type: :utc_datetime_usec)
  end
end
