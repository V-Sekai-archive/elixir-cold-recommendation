defmodule RecGPT.Steam.CanonicalItemText do
  @moduledoc """
  Build canonical item text strings from item_text_dict.pkl so they match the
  RecGPT official pipeline byte-for-byte: Python's str(dict).replace('{','').replace('}','').

  Preserves pkl key order and nested dict key order. Output is a list of binaries (UTF-8)
  in row order (index i = item id i) for use with --canonical-texts. Stored as BLOB in
  SQLite so bytes are not lost to unicode conversion.
  """

  @doc """
  Loads item_text_dict.pkl and returns a list of binaries in row order: [bin_id_0, bin_id_1, ...].
  Each binary is the exact string the official script would pass to the encoder.
  """
  @spec build_ordered_list(String.t()) :: [binary()]
  def build_ordered_list(pkl_path) do
    binary = File.read!(pkl_path)
    {root, _rest} = Unpickler.load!(binary)
    id_to_text = build_id_to_text(root)
    case Map.keys(id_to_text) do
      [] -> []
      keys -> max_id = Enum.max(keys); for id <- 0..max_id, do: Map.get(id_to_text, id, "")
    end
  end

  defp build_id_to_text(%Unpickler.Object{} = root) do
    if root.set_items != [] do
      root.set_items
      |> Enum.map(fn {id_obj, value_obj} ->
        id = unwrap_id(id_obj)
        text = python_repr_dict_replace(value_obj)
        {id, text}
      end)
      |> Map.new()
    else
      # Root may have data in args (e.g. single dict); unwrap and build from map.
      build_id_to_text_from_map(unwrap_object(root))
    end
  end

  defp build_id_to_text(m) when is_map(m), do: build_id_to_text_from_map(m)
  defp build_id_to_text(_), do: %{}

  defp build_id_to_text_from_map(m) when is_map(m) do
    m
    |> Enum.map(fn {id, value_obj} ->
      id = unwrap_id(id)
      text = python_repr_dict_replace(value_obj)
      {id, text}
    end)
    |> Map.new()
  end

  defp build_id_to_text_from_map(_), do: %{}

  defp unwrap_id(obj) do
    case unwrap_object(obj) do
      n when is_integer(n) -> n
      s when is_binary(s) ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> 0
        end
      _ -> 0
    end
  end

  # Python str(dict).replace('{','').replace('}','') — iterate nested dict in order, no outer braces.
  defp python_repr_dict_replace(%Unpickler.Object{} = obj) do
    if obj.set_items != [] do
      obj.set_items
      |> Enum.map(fn {k, v} ->
        k_str = to_string(unwrap_object(k))
        v_str = to_string(unwrap_object(v))
        python_repr_str(k_str) <> ": " <> python_repr_str(v_str)
      end)
      |> Enum.join(", ")
    else
      ""
    end
  end

  defp python_repr_dict_replace(m) when is_map(m) do
    # Fallback when already unwrapped (order lost); deterministic by sorting keys.
    m
    |> Enum.sort_by(fn {k, _} -> to_string(k) end)
    |> Enum.map_join(", ", fn {k, v} ->
      python_repr_str(to_string(k)) <> ": " <> python_repr_str(to_string(v))
    end)
  end

  defp python_repr_dict_replace(_), do: ""

  # Python repr for a string: single quotes around, escape \\ and '.
  defp python_repr_str(s) when is_binary(s) do
    "'" <> escape_python_repr(s) <> "'"
  end

  defp python_repr_str(other), do: python_repr_str(to_string(other))

  defp escape_python_repr(s) do
    s
    |> String.replace("\\", "\\\\")
    |> String.replace("'", "\\'")
  end

  defp unwrap_object(%Unpickler.Object{} = obj) do
    if obj.set_items != [] do
      Map.new(obj.set_items, fn {k, v} -> {unwrap_object(k), unwrap_object(v)} end)
    else
      case obj.args do
        [arg] -> unwrap_object(arg)
        _ -> obj
      end
    end
  end

  defp unwrap_object(s) when is_binary(s), do: s
  defp unwrap_object(n) when is_integer(n), do: n
  defp unwrap_object(f) when is_float(f), do: f
  defp unwrap_object(other), do: other

  @doc """
  Dumps the ordered list of canonical text binaries to the repo's canonical_item_texts table.
  Clears existing rows first. Bytes stored as BLOB so unicode is not applied.
  """
  @spec dump_to_repo(module(), [binary()]) :: :ok
  def dump_to_repo(repo, ordered_list) do
    repo.delete_all(RecGPT.Catalog.CanonicalItemText)
    entries =
      ordered_list
      |> Enum.with_index(0)
      |> Enum.map(fn {text, item_id} -> %{item_id: item_id, text: text} end)
    repo.insert_all(RecGPT.Catalog.CanonicalItemText, entries)
    :ok
  end

  @doc """
  Loads canonical item texts from the repo in row order (item_id 0, 1, 2, ...).
  Returns a list of binaries. Empty list if table is empty.
  """
  @spec load_from_repo(module()) :: [binary()]
  def load_from_repo(repo) do
    import Ecto.Query
    repo.all(
      from c in RecGPT.Catalog.CanonicalItemText,
        order_by: [asc: c.item_id],
        select: c.text
    )
  end
end
