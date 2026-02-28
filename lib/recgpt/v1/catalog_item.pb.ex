defmodule Recgpt.V1.CatalogItem do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.CatalogItem"

  field :item_id, 1, type: :string
  field :slug, 2, type: :string
  field :content_jsonld, 3, type: :string
  field :catalog_ids, 4, repeated: true, type: :string
end
