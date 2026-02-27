defmodule RecGPT.Xmp.DublinCore do
  @moduledoc """
  Dublin Core and DCMI term IRIs for XMP JSON-LD.
  See: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/
  """

  @dc "http://purl.org/dc/elements/1.1/"
  @dcterms "http://purl.org/dc/terms/"

  def dc_title, do: RDF.iri(@dc <> "title")
  def dc_description, do: RDF.iri(@dc <> "description")
  def dc_identifier, do: RDF.iri(@dc <> "identifier")
  def dcterms_source, do: RDF.iri(@dcterms <> "source")

  @doc "Context map for XMP JSON-LD compaction (dc and dcterms prefixes)."
  def context do
    %{
      "dc" => @dc,
      "dcterms" => @dcterms
    }
  end
end
