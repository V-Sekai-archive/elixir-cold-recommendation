defmodule RecGPT.Clickstream.CatalogItem do
  @moduledoc """
  ETNF catalog item: (item_id, source_dataset) -> dc_title, dc_description, etc.
  Dublin Core metadata; XMP JSON-LD is generated from catalog_item via RecGPT.Xmp.Jsonld.from_catalog_item/1.
  """
  use Ecto.Schema

  @primary_key false
  schema "catalog_item" do
    field(:item_id, :integer)
    field(:source_dataset, :string)
    field(:dc_title, :string)
    field(:dc_description, :string)
    field(:dc_type, :string)
    field(:dcterms_source, :string)
    timestamps(type: :utc_datetime_usec)
  end
end
