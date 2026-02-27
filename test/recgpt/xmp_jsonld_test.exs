defmodule RecGPT.XmpJsonldTest do
  use ExUnit.Case, async: true

  alias RecGPT.Xmp.DublinCore
  alias RecGPT.Xmp.Jsonld

  describe "from_catalog_item/1" do
    test "builds XMP JSON-LD string from catalog item map" do
      row = %{
        item_id: 42,
        source_dataset: "uci_clickstream",
        dc_title: "category 1 product 42",
        dc_description: "category 1 product 42 colour blue",
        dcterms_source: "uci_clickstream"
      }

      assert {:ok, jsonld} = Jsonld.from_catalog_item(row)
      assert is_binary(jsonld)

      parsed = Jason.decode!(jsonld)
      assert parsed["@context"]["dc"] == "http://purl.org/dc/elements/1.1/"
      assert parsed["@context"]["dcterms"] == "http://purl.org/dc/terms/"
      assert parsed["dc:title"] == row.dc_title
      assert parsed["dc:description"] == row.dc_description
      assert parsed["dc:identifier"] == "uci_clickstream:42"
      assert parsed["dcterms:source"] == row.dcterms_source
    end

    test "defaults dcterms_source to source_dataset when missing" do
      row = %{
        item_id: 0,
        source_dataset: "my_dataset",
        dc_title: "Title",
        dc_description: "Desc"
      }

      assert {:ok, jsonld} = Jsonld.from_catalog_item(row)
      parsed = Jason.decode!(jsonld)
      assert parsed["dcterms:source"] == "my_dataset"
      assert parsed["dc:identifier"] == "my_dataset:0"
    end

    test "returns error for invalid input" do
      assert Jsonld.from_catalog_item(%{}) == {:error, :invalid_catalog_item}
      assert Jsonld.from_catalog_item(%{item_id: 1}) == {:error, :invalid_catalog_item}
    end
  end

  describe "validate_jsonld/1" do
    test "validates JSON-LD produced by from_catalog_item" do
      row = %{
        item_id: 1,
        source_dataset: "test",
        dc_title: "A",
        dc_description: "B",
        dcterms_source: "test"
      }

      {:ok, jsonld} = Jsonld.from_catalog_item(row)
      assert :ok == Jsonld.validate_jsonld(jsonld)
    end

    test "validates example from docs (04)" do
      example = """
      {
        "@context": {
          "dc": "http://purl.org/dc/elements/1.1/",
          "dcterms": "http://purl.org/dc/terms/"
        },
        "dc:title": "category 1 product 42 colour blue",
        "dc:description": "category 1 product 42 colour blue",
        "dc:identifier": "uci_clickstream:42",
        "dcterms:source": "uci_clickstream"
      }
      """

      assert :ok == Jsonld.validate_jsonld(example)
    end

    test "returns error for invalid JSON-LD" do
      assert {:error, _} = Jsonld.validate_jsonld("not json")
      assert {:error, _} = Jsonld.validate_jsonld("{}")
    end
  end

  describe "to_xmp_jsonld_string/2" do
    test "accepts context and pretty options" do
      alias RecGPT.Xmp.CatalogItemSchema, as: Schema

      struct =
        Schema.build!(RDF.bnode(), title: "T", description: "D", identifier: "i:1", source: "s")

      {:ok, graph} = Grax.to_rdf(struct)

      assert {:ok, _} =
               Jsonld.to_xmp_jsonld_string(graph, context: DublinCore.context(), pretty: true)
    end
  end
end
