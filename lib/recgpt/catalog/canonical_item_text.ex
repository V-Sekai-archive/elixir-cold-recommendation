defmodule RecGPT.Catalog.CanonicalItemText do
  @moduledoc "ETNF: canonical_item_texts(item_id PK, text BLOB) — RecGPT-official bytes per item."
  use Ecto.Schema

  @primary_key {:item_id, :integer, autogenerate: false}
  schema "canonical_item_texts" do
    field(:text, :binary)
  end
end
