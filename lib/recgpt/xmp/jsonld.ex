defmodule RecGPT.Xmp.Jsonld do
  @moduledoc """
  Enforces XMP JSON-LD for catalog items using RDF.ex and Grax.

  Flow: RDBMS (catalog_item) → Grax struct → RDF graph → JSON-LD (compact with DC context).

  Uses [RDF.ex](https://rdf-elixir.dev/) and [Grax](https://rdf-elixir.dev/grax/) so that:
  - Catalog rows are mapped to a schema-conformant Grax struct (Dublin Core).
  - The struct is serialized to RDF and then to compact XMP JSON-LD with `@context`.
  - Existing JSON-LD can be validated by parsing → RDF → Grax.load.
  """

  alias RecGPT.Xmp.CatalogItemSchema, as: Schema
  alias RecGPT.Xmp.DublinCore, as: DC

  @doc """
  Builds XMP JSON-LD string from a catalog item (Ecto struct or map).

  Expects `:item_id`, `:source_dataset`, `:dc_title`, `:dc_description`, and optionally
  `:dcterms_source` (defaults to `source_dataset`). Converts to Grax struct, then RDF,
  then compact JSON-LD with Dublin Core context.

  Returns `{:ok, jsonld_string}` or `{:error, reason}`.
  """
  def from_catalog_item(%{item_id: _id, source_dataset: _src} = row) do
    identifier = "#{row.source_dataset}:#{row.item_id}"
    source = Map.get(row, :dcterms_source) || row.source_dataset

    struct =
      Schema.build!(RDF.bnode(),
        title: row.dc_title,
        description: row.dc_description,
        identifier: identifier,
        source: source
      )

    struct
    |> Grax.to_rdf()
    |> case do
      {:ok, graph} -> to_xmp_jsonld_string(graph)
      {:error, _} = err -> err
    end
  end

  def from_catalog_item(_), do: {:error, :invalid_catalog_item}

  @doc """
  Serializes an RDF graph or dataset to XMP-style JSON-LD string using Dublin Core context.
  """
  def to_xmp_jsonld_string(rdf_data, opts \\ []) do
    context = Keyword.get(opts, :context, DC.context())
    pretty = Keyword.get(opts, :pretty, false)
    JSON.LD.write_string(rdf_data, context: context, pretty: pretty)
  end

  @doc """
  Validates a JSON-LD string against the catalog item Grax schema.

  Parses JSON-LD → RDF, then loads the first subject into `RecGPT.Xmp.CatalogItemSchema`.
  Returns `:ok` if load (and optional validation) succeeds, otherwise `{:error, reason}`.
  """
  def validate_jsonld(jsonld_string) do
    case JSON.LD.read_string(jsonld_string) do
      {:ok, data} -> validate_jsonld_data(data)
      {:error, _} = err -> err
    end
  end

  defp validate_jsonld_data(data) do
    graph = to_graph(data)
    subjects = data |> RDF.Data.subjects() |> Enum.to_list()

    case subjects do
      [subject | _] -> Grax.load(graph, subject, Schema) |> ok_or_error()
      [] -> {:error, :no_subject}
    end
  end

  defp ok_or_error({:ok, _}), do: :ok
  defp ok_or_error({:error, _} = err), do: err

  defp to_graph(%RDF.Graph{} = g), do: g
  defp to_graph(%RDF.Dataset{graphs: graphs}), do: Map.get(graphs, nil) || RDF.Graph.new()
end
