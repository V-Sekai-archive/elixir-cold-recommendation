defmodule RecGPT.Serve do
  @moduledoc """
  Next-item recommendation and catalog search (backend for REST API).

  Loads model + token_id_list + trie once. The HTTP surface is a RESTful API
  (Google API Design Guide): see `RecGPT.Serve.Plug` and `RecGPT.Serve.REST`.
  Run: mix recgpt.serve [--port 8000]. Requires fixture and checkpoint export dir.
  """

  alias RecGPT.CheckpointLoader
  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.Trie

  @padding_id 15_360
  @max_length 255
  @seq_token_capacity 1024

  defstruct [:params, :trie, :token_id_list, :item_text, :num_items, :get_logits_fn, :item_extra]

  @type state :: %__MODULE__{
          params: map(),
          trie: map(),
          token_id_list: [[non_neg_integer()]],
          item_text: %{non_neg_integer() => String.t() | map()},
          num_items: non_neg_integer(),
          get_logits_fn: (list(non_neg_integer()) -> Nx.Tensor.t()),
          item_extra:
            nil
            | %{optional(non_neg_integer()) => map()}
            | (non_neg_integer(), %__MODULE__{} -> map())
        }

  @doc """
  Load server state: checkpoint export, fixture (token_id_list), optional catalog JSON.

  Options:
  - :item_extra — Optional. Map or function to add extra fields per item in API responses.
    - Map: %{item_id => %{"asset_id" => "...", "slug" => "..."}} (e.g. for Polymarket).
    - Function: (item_id, state) -> %{"key" => "value"}.
  Returns {:ok, state} or {:error, reason}.
  """
  def load_state(fixture_path, ckpt_export_dir, catalog_path \\ nil, opts \\ []) do
    with {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {:ok, token_id_list, num_items} <- load_fixture(fixture_path),
         {:ok, item_text} <- load_catalog(catalog_path, num_items) do
      trie = Trie.build(token_id_list)
      get_logits_fn = build_get_logits_fn(params)
      item_extra = Keyword.get(opts, :item_extra)

      state = %__MODULE__{
        params: params,
        trie: trie,
        token_id_list: token_id_list,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn,
        item_extra: item_extra
      }

      {:ok, state}
    end
  end

  defp load_checkpoint(dir) do
    manifest = Path.join(dir, "manifest.json")

    if File.regular?(manifest) do
      {:ok, CheckpointLoader.load_from_export(dir)}
    else
      {:error, "checkpoint not found: #{dir}"}
    end
  end

  defp load_fixture(path) do
    if File.regular?(path) do
      fixture = File.read!(path) |> Jason.decode!()

      token_id_list =
        (fixture["token_id_list"] || []) |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

      num_items = fixture["num_items"] || length(token_id_list)
      {:ok, token_id_list, num_items}
    else
      {:error, "fixture not found: #{path}"}
    end
  end

  defp load_catalog(nil, _num_items), do: {:ok, %{}}
  defp load_catalog(path, _num_items) when path in [nil, "", []], do: {:ok, %{}}

  defp load_catalog(path, _num_items) do
    if File.regular?(path) do
      raw = File.read!(path) |> Jason.decode!()

      item_text =
        case raw do
          %{"items" => items} when is_list(items) ->
            Enum.reduce(items, %{}, fn item, acc ->
              id = item["id"] || item["item_id"]
              text = item["text"] || item["title"] || item["raw"] || ""
              if is_integer(id), do: Map.put(acc, id, text), else: acc
            end)

          map when is_map(map) ->
            Enum.reduce(map, %{}, fn {k, v}, acc ->
              case Integer.parse(to_string(k)) do
                {id, _} -> Map.put(acc, id, v)
                :error -> acc
              end
            end)
        end

      {:ok, item_text}
    else
      {:ok, %{}}
    end
  end

  defp build_get_logits_fn(params) do
    fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end
  end

  @doc """
  Convert item_ids (catalog indices) to left-padded token sequence for inference (same as Python serve seq_to_batch).
  """
  def item_ids_to_context_token_ids(item_ids, token_id_list, padding_id \\ @padding_id) do
    seq = Enum.take(item_ids, -@max_length)
    token_list = Enum.flat_map(seq, fn iid -> Enum.at(token_id_list, iid) || [0, 0, 0, 0] end)
    len = length(token_list)
    padding = List.duplicate(padding_id, @seq_token_capacity - len)
    padding ++ token_list
  end

  @doc """
  Recommend next item(s) given context item_ids. Returns up to `top_k` item_ids (best first) from beam search.
  """
  def recommend(state, item_ids, top_k \\ 5)
      when is_list(item_ids) and is_integer(top_k) and top_k >= 1 do
    if item_ids == [] do
      {:error, "item_ids must be non-empty"}
    else
      top_k = min(top_k, 20)
      context_token_ids = item_ids_to_context_token_ids(item_ids, state.token_id_list)

      case Decode.beam_search_top_k(state.get_logits_fn, state.trie, context_token_ids, top_k) do
        {:ok, list} -> {:ok, list}
        :not_found -> {:ok, []}
      end
    end
  end

  @doc """
  Search catalog by string q; returns list of %{item_id: id, ...} up to limit.
  """
  def search(state, q, limit \\ 20) do
    q = (q || "") |> String.downcase() |> String.trim()

    if q == "" or state.item_text == %{} do
      []
    else
      state.item_text
      |> Enum.filter(fn {_id, text} ->
        str = if is_map(text), do: inspect(text), else: to_string(text)
        String.contains?(String.downcase(str), q)
      end)
      |> Enum.take(limit)
      |> Enum.map(fn {id, text} ->
        %{"item_id" => id, "raw" => safe_str(text)}
      end)
    end
  end

  @doc "Coerce value to string for API response (nil -> \"\", binary -> self, map -> inspect, other -> to_string)."
  def safe_str(nil), do: ""
  def safe_str(s) when is_binary(s), do: s
  def safe_str(m) when is_map(m), do: inspect(m)
  def safe_str(x), do: to_string(x)
end
