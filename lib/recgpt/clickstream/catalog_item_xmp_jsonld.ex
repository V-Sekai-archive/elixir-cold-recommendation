defmodule RecGPT.Clickstream.CatalogItemXmpJsonld do
  @moduledoc """
  Read-only view: catalog_item_xmp_jsonld.
  Exposes item_id, source_dataset, and item_xmp_jsonld (Dublin Core XMP JSON-LD as TEXT).
  Do not insert/update; query only.
  """
  use Ecto.Schema

  @primary_key false
  schema "catalog_item_xmp_jsonld" do
    field :item_id, :integer
    field :source_dataset, :string
    field :item_xmp_jsonld, :string
  end
end
