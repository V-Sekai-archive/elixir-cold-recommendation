defmodule RecGPT.Xmp.CatalogItemSchema do
  @moduledoc """
  Grax schema for a single catalog item as Dublin Core XMP JSON-LD.
  Maps RDF graph ↔ struct for validation and RDBMS → JSON-LD enforcement.
  """
  use Grax.Schema

  alias RecGPT.Xmp.DublinCore, as: DC

  schema do
    property(:title, DC.dc_title(), type: :string)
    property(:description, DC.dc_description(), type: :string)
    property(:identifier, DC.dc_identifier(), type: :string)
    property(:source, DC.dcterms_source(), type: :string)
  end
end
