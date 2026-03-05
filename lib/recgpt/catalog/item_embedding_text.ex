defmodule RecGPT.Catalog.ItemEmbeddingText do
  @moduledoc "ETNF: item_embedding_texts(item_id PK, embedding_text). Text used as embedding input; when present, use it; else fall back to items.title."
  use Ecto.Schema

  @primary_key {:item_id, :integer, autogenerate: false}
  schema "item_embedding_texts" do
    field(:embedding_text, :string)
  end
end
