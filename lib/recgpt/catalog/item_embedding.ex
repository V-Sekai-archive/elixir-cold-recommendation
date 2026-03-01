defmodule RecGPT.Catalog.ItemEmbedding do
  @moduledoc "ETNF: item_embeddings(item_id, embedding BLOB)."
  use Ecto.Schema

  @primary_key {:item_id, :integer, autogenerate: false}
  schema "item_embeddings" do
    field(:embedding, :binary)
  end
end
