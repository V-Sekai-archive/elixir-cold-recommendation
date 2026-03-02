defmodule Mix.Tasks.Recgpt.InspectItemText do
  @shortdoc "Inspect item_text_dict.pkl and items.json text format (for embedding parity)"
  @moduledoc """
  Loads item_text_dict.pkl (and items.json if present) and prints the first N raw values
  so we can see the exact text shape the reference may have used for item_text_embeddings.npy.

  Run after: mix recgpt.fetch_steam data/steam

  ## Options
    * `--steam-dir` - Directory with item_text_dict.pkl (default: data/steam)
    * `--limit` - Number of items to print (default: 5)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [steam_dir: :string, limit: :integer])

    dir = opts[:steam_dir] || Path.expand("data/steam", File.cwd!())
    limit = opts[:limit] || 5
    pkl_path = Path.join(dir, "item_text_dict.pkl")
    items_path = Path.join(dir, "items.json")

    unless File.regular?(pkl_path) do
      Mix.raise("item_text_dict.pkl not found at #{pkl_path}. Run mix recgpt.fetch_steam first.")
    end

    Mix.shell().info("Loading #{pkl_path}...")
    raw = load_pkl(pkl_path)
    map = to_map(raw)
    keys = map |> Map.keys() |> Enum.sort_by(&sort_key/1) |> Enum.take(limit)

    Mix.shell().info("")
    Mix.shell().info("=== item_text_dict.pkl: first #{length(keys)} values (raw) ===")

    Mix.shell().info(
      "Reference likely encoded these strings to produce item_text_embeddings.npy."
    )

    Mix.shell().info("")

    for {idx, k} <- Enum.with_index(keys) do
      v = map[k] || map[to_string(k)]
      {type_str, preview} = value_preview(v)
      Mix.shell().info("  [#{idx}] key=#{inspect(k)}  type=#{type_str}")
      Mix.shell().info("       preview: #{preview}")
      Mix.shell().info("")
    end

    if File.regular?(items_path) do
      Mix.shell().info("=== items.json: first #{limit} titles (what we use) ===")
      raw_json = File.read!(items_path) |> Jason.decode!()
      items = (raw_json["items"] || []) |> Enum.take(limit)

      for {item, idx} <- Enum.with_index(items) do
        title = item["title"] || ""
        recgpt_style = RecGPT.Embedding.recgpt_item_text(item)
        Mix.shell().info("  [#{idx}] title=#{inspect(truncate(title, 60))}")
        Mix.shell().info("       recgpt_item_text(item)=#{inspect(truncate(recgpt_style, 80))}")
        Mix.shell().info("")
      end
    end
  end

  defp load_pkl(path) do
    binary = File.read!(path)
    {root, _rest} = Unpickler.load!(binary)
    unwrap_object(root)
  end

  defp unwrap_object(%Unpickler.Object{} = obj) do
    if obj.set_items != [] do
      Map.new(obj.set_items, fn {k, v} -> {unwrap_object(k), unwrap_object(v)} end)
    else
      case obj.args do
        [m] when is_map(m) -> m
        _ -> obj
      end
    end
  end

  defp unwrap_object(other), do: other

  defp to_map(%Unpickler.Object{} = o), do: unwrap_object(o)
  defp to_map(m) when is_map(m), do: m
  defp to_map(_), do: %{}

  defp sort_key(x) when is_integer(x), do: {0, x}

  defp sort_key(x) when is_binary(x) do
    case Integer.parse(x) do
      {n, _} -> {0, n}
      :error -> {1, x}
    end
  end

  defp sort_key(x), do: {2, inspect(x)}

  defp value_preview(v) when is_binary(v) do
    {"string", truncate(v, 120)}
  end

  defp value_preview(v) when is_map(v) do
    # Could be %{"title" => "X"} or already unwrapped dict
    title = Map.get(v, "title") || Map.get(v, :title)

    if title != nil do
      {"map (has title)", "title => #{truncate(to_string(title), 80)}"}
    else
      keys = Enum.map_join(Map.keys(v), ", ", &inspect/1)
      {"map", "keys: #{keys}"}
    end
  end

  defp value_preview(v) do
    s = inspect(v)
    type = if is_map(v) and Map.has_key?(v, :__struct__), do: inspect(v.__struct__), else: "other"
    {"#{type}", truncate(s, 120)}
  end

  defp truncate(s, len) when is_binary(s) do
    if String.length(s) <= len, do: s, else: String.slice(s, 0, len) <> "..."
  end

  defp truncate(s, len), do: truncate(to_string(s), len)
end
