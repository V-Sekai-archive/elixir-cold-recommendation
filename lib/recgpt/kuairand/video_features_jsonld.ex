defmodule RecGPT.KuaiRand.VideoFeaturesJsonld do
  @moduledoc """
  Builds canonical JSON-LD (XMP-style) text per video from KuaiRand-Pure item features.

  Uses RFC 8785 (JSON Canonicalization Scheme, JCS): object keys sorted lexicographically
  (UTF-16 code unit order), recursive, no unnecessary whitespace. Reads
  `video_features_basic_pure.csv` and `video_features_statistic_pure.csv`, joins by
  video_id, emits one canonical JSON string per video. Suitable for embedding (canonical
  item text or item_embedding_text).
  """

  @jsonld_context "https://schema.org"
  @jsonld_type "VideoObject"

  @doc """
  Loads both video feature CSVs from `dir`, joins by video_id, returns a map
  `video_id (integer) -> canonical JSON-LD text string`.

  Options:
    * `:basic_path` - Override path to video_features_basic_pure.csv
    * `:stat_path` - Override path to video_features_statistic_pure.csv
  """
  @spec load_canonical_texts(String.t(), keyword()) :: {:ok, %{optional(integer()) => String.t()}} | {:error, term()}
  def load_canonical_texts(dir, opts \\ []) do
    dir = Path.expand(dir)
    basic_path = opts[:basic_path] || Path.join(dir, "video_features_basic_pure.csv")
    stat_path = opts[:stat_path] || Path.join(dir, "video_features_statistic_pure.csv")

    with {:ok, basic_rows} <- parse_csv_file(basic_path),
         {:ok, stat_rows} <- parse_csv_file(stat_path),
         joined <- join_by_video_id(basic_rows, stat_rows) do
      by_video =
        joined
        |> Enum.map(&row_to_jsonld_object/1)
        |> Enum.map(&canonicalize_jsonld/1)
        |> Map.new(fn {vid, json} -> {vid, json} end)

      {:ok, by_video}
    end
  rescue
    e -> {:error, e}
  end

  @doc """
  Same as load_canonical_texts/2 but returns a list of canonical text strings in
  **item index order** (sorted video_id), so index i matches items.json item id i.
  """
  @spec load_canonical_texts_ordered(String.t(), keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def load_canonical_texts_ordered(dir, opts \\ []) do
    case load_canonical_texts(dir, opts) do
      {:ok, by_video} ->
        sorted_ids = by_video |> Map.keys() |> Enum.sort()
        list = Enum.map(sorted_ids, &Map.fetch!(by_video, &1))
        {:ok, list}

      err ->
        err
    end
  end

  defp parse_csv_file(path) do
    unless File.regular?(path), do: raise("File not found: #{path}")
    content = File.read!(path)
    rows = NimbleCSV.RFC4180.parse_string(content, skip_headers: false)
    [header | data] = rows
    header = Enum.map(header, &String.downcase(String.trim(&1)))
    col_idx = Enum.with_index(header) |> Map.new(fn {name, i} -> {name, i} end)

    get = fn row, name ->
      idx = Map.get(col_idx, name)
      if idx != nil, do: Enum.at(row, idx), else: nil
    end

    parsed =
      Enum.map(data, fn row ->
        Map.new(header, fn name ->
          val = get.(row, name)
          {name, if(is_binary(val), do: String.trim(val), else: val)}
        end)
      end)

    {:ok, parsed}
  end

  defp join_by_video_id(basic_rows, stat_rows) do
    by_video_basic =
      basic_rows
      |> Enum.filter(fn r -> parse_int(r["video_id"]) != nil end)
      |> Map.new(fn r -> {parse_int(r["video_id"]), r} end)

    by_video_stat =
      stat_rows
      |> Enum.filter(fn r -> parse_int(r["video_id"]) != nil end)
      |> Map.new(fn r -> {parse_int(r["video_id"]), r} end)

    video_ids = MapSet.union(MapSet.new(Map.keys(by_video_basic)), MapSet.new(Map.keys(by_video_stat))) |> MapSet.to_list() |> Enum.sort()

    Enum.map(video_ids, fn vid ->
      b = Map.get(by_video_basic, vid, %{})
      s = Map.get(by_video_stat, vid, %{})
      # Merge: basic first, then stat (stat keys override if duplicate)
      Map.merge(b, s)
    end)
  end

  defp row_to_jsonld_object(row) do
    video_id = parse_int(row["video_id"])
    # JSON-LD / XMP: @context, @type, @id first, then all other properties (sorted)
    base = %{
      "@context" => @jsonld_context,
      "@type" => @jsonld_type,
      "@id" => "video_id:#{video_id}"
    }

    props =
      row
      |> Map.delete("video_id")
      |> Enum.map(fn {k, v} -> {k, coerce_value(v)} end)
      |> Enum.reject(fn {_, v} -> v == nil end)
      |> Map.new()

    obj = Map.merge(base, props)
    {video_id, obj}
  end

  defp coerce_value(""), do: nil
  defp coerce_value(nil), do: nil

  defp coerce_value(v) when is_binary(v) do
    v = String.trim(v)
    case Integer.parse(v) do
      {n, ""} -> n
      _ -> case Float.parse(v) do
             {f, ""} -> f
             _ -> v
           end
    end
  end

  defp coerce_value(v), do: v

  defp parse_int(nil), do: nil
  defp parse_int(s) when is_binary(s) do
    case Integer.parse(String.trim(s)) do
      {n, _} -> n
      :error -> nil
    end
  end
  defp parse_int(n) when is_integer(n), do: n
  defp parse_int(_), do: nil

  # RFC 8785 (JCS) via the jcs library.
  defp canonicalize_jsonld({video_id, obj}) do
    json = Jcs.encode(obj)
    {video_id, json}
  end
end
