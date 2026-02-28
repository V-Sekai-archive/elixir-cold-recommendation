defmodule RecGPT.Steam.Fetch do
  @moduledoc """
  Download Steam test split from HuggingFace (hkuds/RecGPT_dataset) and write
  recgpt JSON artifacts: items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json.

  Uses Unpickler to parse Python pickle files (no Python required).
  """

  @base_url "https://huggingface.co/datasets/hkuds/RecGPT_dataset/resolve/main/test/steam"
  @max_context 64

  @doc """
  Downloads pkl files, unpickles them, and writes JSON to `out_dir`.
  Returns `:ok` or `{:error, reason}`.

  Options:
  - `:out_dir` — output directory (default: "data/steam")
  """
  def run(out_dir \\ "data/steam", _opts \\ []) do
    dir = Path.expand(out_dir, File.cwd!())
    File.mkdir_p!(dir)
    Application.ensure_all_started(:req)

    with :ok <- ensure_item_text_dict(dir),
         {:ok, item_ids, old_to_new, title_map} <- build_item_map(dir),
         :ok <- write_items_json(dir, item_ids, title_map) do
      write_sequence_jsons(dir, old_to_new)
    end
  end

  defp ensure_item_text_dict(dir) do
    path = Path.join(dir, "item_text_dict.pkl")
    if File.regular?(path), do: :ok, else: download_pkl(@base_url, "item_text_dict.pkl", path)
  end

  defp download_pkl(base_url, filename, path) do
    url = "#{base_url}/#{filename}"
    Mix.shell().info("Downloading #{filename}...")

    case Req.get(url) do
      {:ok, %{status: 200, body: body}} ->
        File.write!(path, body)
        :ok
      {:ok, %{status: code}} ->
        {:error, "HTTP #{code} for #{url}"}
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp load_pkl(path) do
    binary = File.read!(path)
    {root, _rest} = Unpickler.load!(binary)
    unwrap_object(root)
  end

  defp unwrap_object(%Unpickler.Object{} = obj) do
    # OrderedDict or dict-like: may have set_items as list of {k, v}
    if obj.set_items != [] do
      Map.new(obj.set_items)
    else
      # Fallback: try args (e.g. single dict arg)
      case obj.args do
        [m] when is_map(m) -> m
        _ -> obj
      end
    end
  end
  defp unwrap_object(other), do: other

  defp build_item_map(dir) do
    path = Path.join(dir, "item_text_dict.pkl")
    raw = load_pkl(path)
    map = to_map(raw)
    keys = map |> Map.keys() |> Enum.sort_by(&sort_key/1)
    old_to_new = keys |> Enum.with_index() |> Map.new(fn {old, i} -> {old, i} end)
    title_map = Map.new(keys, fn k -> {k, to_string(map[k] || map[to_string(k)] || "")} end)
    {:ok, keys, old_to_new, title_map}
  end

  defp sort_key(x) when is_integer(x), do: {0, x}
  defp sort_key(x) when is_binary(x) do
    case Integer.parse(x) do
      {n, _} -> {0, n}
      :error -> {1, x}
    end
  end
  defp sort_key(x), do: {2, inspect(x)}

  defp to_map(%Unpickler.Object{} = o), do: unwrap_object(o)
  defp to_map(m) when is_map(m), do: m
  defp to_map(_), do: %{}

  defp write_items_json(dir, item_ids, title_map) do
    items =
      Enum.map(Enum.with_index(item_ids), fn {old_id, i} ->
        %{"id" => i, "title" => title_map[old_id] || ""}
      end)

    num_items = length(items)
    out = Path.join(dir, "items.json")
    File.write!(out, Jason.encode!(%{"items" => items, "num_items" => num_items}, pretty: true))
    Mix.shell().info("Wrote #{out}")
    :ok
  end

  defp write_sequence_jsons(dir, old_to_new) do
    num_items = map_size(old_to_new)

    for fname <- ["train.pkl", "cold_train.pkl"] do
      path = Path.join(dir, fname)
      ensure_pkl!(dir, fname, path)
      raw = load_pkl(path)
      seqs = to_list_of_lists(raw)
      mapped = Enum.map(seqs, fn seq -> map_sequence(seq, old_to_new) end)
      out_name = fname |> String.replace(".pkl", "_sequences") |> then(& &1 <> ".json")
      out_path = Path.join(dir, out_name)
      File.write!(out_path, Jason.encode!(%{"sequences" => mapped, "num_items" => num_items}, pretty: true))
      Mix.shell().info("Wrote #{out_path} (#{length(mapped)} sequences)")
    end

    for fname <- ["test.pkl", "cold_test.pkl"] do
      path = Path.join(dir, fname)
      ensure_pkl!(dir, fname, path)
      raw = load_pkl(path)
      seqs = to_list_of_lists(raw)
      test_cases = Enum.map(seqs, fn seq -> seq_to_test_case(map_sequence(seq, old_to_new)) end)
      out_name = fname |> String.replace(".pkl", "_sequences") |> then(& &1 <> ".json")
      out_path = Path.join(dir, out_name)
      File.write!(out_path, Jason.encode!(%{"test_cases" => test_cases, "num_items" => map_size(old_to_new)}, pretty: true))
      Mix.shell().info("Wrote #{out_path} (#{length(test_cases)} test cases)")
    end

    :ok
  end

  defp ensure_pkl!(_dir, filename, path) do
    unless File.regular?(path) do
      case download_pkl(@base_url, filename, path) do
        :ok -> :ok
        {:error, reason} -> raise "Download failed: #{inspect(reason)}"
      end
    end
  end

  defp to_list_of_lists(%Unpickler.Object{} = o) do
    # Could be dict values
    unwrapped = unwrap_object(o)
    if is_map(unwrapped), do: Map.values(unwrapped), else: List.wrap(unwrapped)
  end
  defp to_list_of_lists(m) when is_map(m), do: Map.values(m)
  defp to_list_of_lists(l) when is_list(l), do: l
  defp to_list_of_lists(_), do: []

  defp map_sequence(seq, old_to_new) when is_list(seq) do
    Enum.map(seq, fn x -> Map.get(old_to_new, x, x) end)
  end
  defp map_sequence(_, _), do: []

  defp seq_to_test_case([]), do: %{"context" => [], "next_item" => 0}
  defp seq_to_test_case([single]), do: %{"context" => [], "next_item" => single}
  defp seq_to_test_case(seq) do
    context = seq |> Enum.drop(-1) |> Enum.take(-@max_context)
    next_item = List.last(seq)
    %{"context" => context, "next_item" => next_item}
  end
end
