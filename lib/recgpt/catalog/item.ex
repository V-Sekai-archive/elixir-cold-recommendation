defmodule RecGPT.Catalog.Item do
  @moduledoc "ETNF: items(item_id, title)."
  use Ecto.Schema

  @primary_key {:item_id, :integer, autogenerate: false}
  schema "items" do
    field(:title, :string)
  end
end
