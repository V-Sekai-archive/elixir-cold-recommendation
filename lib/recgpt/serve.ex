defmodule RecGPT.Serve do
  @moduledoc """
  Next-item recommendation (backend for gRPC API).

  Implements `RecGPT.RecommendationService`; used as the default implementation
  when the application calls the recommendation service. Loads model + token_id_list + trie once.
  Served via gRPC (recgpt.v1.PredictionService). Run: mix recgpt.serve [--grpc-port 50051].
  Requires fixture and checkpoint export dir.
  """
  @behaviour RecGPT.RecommendationService

  alias RecGPT.CheckpointLoader
  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.InferenceDefn
  alias RecGPT.InferenceParams
  alias RecGPT.Trie

  @padding_id 15_360
  @max_length 255
  @seq_token_capacity 1024

  defstruct [
    :params,
    :trie,
    :token_id_list,
    :token_id_map,
    :item_text,
    :num_items,
    :get_logits_fn,
    :get_logits_batch_fn
  ]

  @type state :: %__MODULE__{
          params: map(),
          trie: map(),
          token_id_list: [[non_neg_integer()]],
          token_id_map: %{non_neg_integer() => [non_neg_integer()]} | nil,
          item_text: %{non_neg_integer() => String.t() | map()},
          num_items: non_neg_integer(),
          get_logits_fn: (list(non_neg_integer()) -> Nx.Tensor.t()),
          get_logits_batch_fn: ([[non_neg_integer()]] -> Nx.Tensor.t()) | nil
        }

  @doc """
  Load server state: checkpoint export, fixture (token_id_list), optional catalog JSON.
  Returns {:ok, state} or {:error, reason}.
  """
  @spec load_state(String.t(), String.t(), String.t() | nil) ::
          {:ok, state()} | {:error, String.t()}
  def load_state(fixture_path, ckpt_export_dir, catalog_path \\ nil) do
    with :ok <- ensure_torchx(),
         {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {:ok, token_id_list, num_items} <- load_fixture(fixture_path),
         {:ok, item_text} <- load_catalog(catalog_path, num_items),
         {:ok, get_logits_batch_fn} <- build_get_logits_batch_fn(params) do
      trie = Trie.build(token_id_list)
      get_logits_fn = build_get_logits_fn_from_batch(get_logits_batch_fn)

      state = %__MODULE__{
        params: params,
        trie: trie,
        token_id_list: token_id_list,
        token_id_map: nil,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn,
        get_logits_batch_fn: get_logits_batch_fn
      }

      {:ok, state}
    end
  end

  @doc """
  Load state from catalog DB (item_tokens) + checkpoint. Constant memory: streams item_tokens into trie and map.
  Requires RECGPT_SQLITE_PATH and mix ecto.migrate. Run build_fixture with --sqlite (or RECGPT_SQLITE_PATH) first.
  """
  @spec load_state_from_db(String.t(), String.t() | nil) :: {:ok, state()} | {:error, String.t()}
  def load_state_from_db(ckpt_export_dir, catalog_path \\ nil) do
    with :ok <- ensure_torchx(),
         {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {:ok, trie, token_id_map, num_items} <- load_fixture_from_db(),
         {:ok, item_text} <- load_catalog(catalog_path, num_items),
         {:ok, get_logits_batch_fn} <- build_get_logits_batch_fn(params) do
      get_logits_fn = build_get_logits_fn_from_batch(get_logits_batch_fn)

      state = %__MODULE__{
        params: params,
        trie: trie,
        token_id_list: [],
        token_id_map: token_id_map,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn,
        get_logits_batch_fn: get_logits_batch_fn
      }

      {:ok, state}
    end
  end

  defp load_fixture_from_db do
    import Ecto.Query
    alias RecGPT.Catalog.ItemToken
    alias RecGPT.Repo

    stream =
      from(t in ItemToken,
        order_by: [asc: t.item_id],
        select: {t.item_id, t.t0, t.t1, t.t2, t.t3}
      )
      |> Repo.stream()

    stream =
      Stream.map(stream, fn {item_id, t0, t1, t2, t3} ->
        tokens = [t0 || 0, t1 || 0, t2 || 0, t3 || 0]
        {item_id, tokens}
      end)

    {trie, token_id_map, num_items} =
      Enum.reduce(stream, {%{}, %{}, 0}, fn {item_id, tokens}, {acc_trie, acc_map, max_id} ->
        new_trie = Trie.add_item(acc_trie, item_id, tokens)
        new_map = Map.put(acc_map, item_id, tokens)
        new_max = max(item_id + 1, max_id)
        {new_trie, new_map, new_max}
      end)

    {:ok, trie, token_id_map, num_items}
  end

  defp ensure_torchx do
    if Code.ensure_loaded?(Torchx) do
      :ok
    else
      {:error,
       "Torchx required for inference. Add {:torchx, \"~> 0.11\"} to deps and ensure it compiles."}
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

          _ ->
            %{}
        end

      {:ok, item_text}
    else
      {:ok, %{}}
    end
  end

  defp build_get_logits_fn_from_batch(get_logits_batch_fn) do
    fn token_list ->
      {logits, _cache} = get_logits_batch_fn.([token_list], nil)
      Nx.squeeze(logits, axes: [0])
    end
  end

  defp build_get_logits_batch_fn(params) do
    try do
      n_layers = Inference.n_layers_from_params(params)
      full_params = InferenceParams.build_defn_params(params, n_layers)

      # Use Nx.Defn.Evaluator: Torchx is a backend, not a Defn.Compiler (no __jit__/5).
      jit_with_cache =
        Nx.Defn.jit(&InferenceDefn.forward_with_cache/4, compiler: Nx.Defn.Evaluator)

      jit_incremental =
        Nx.Defn.jit(&InferenceDefn.forward_incremental/5, compiler: Nx.Defn.Evaluator)

      batch_fn = fn list_of_token_lists, cache
                    when is_list(list_of_token_lists) and list_of_token_lists != [] ->
        max_len = list_of_token_lists |> Enum.map(&length/1) |> Enum.max()

        padded =
          Enum.map(list_of_token_lists, fn tokens ->
            len = length(tokens)
            padding = List.duplicate(@padding_id, max_len - len)
            padding ++ tokens
          end)

        batch = Nx.tensor(padded, type: {:s, 32})
        {batch_size, seq_len} = Nx.shape(batch)
        batch_aux = Nx.broadcast(0.0, {batch_size, seq_len, 192}) |> Nx.as_type({:f, 32})
        embed_mask = Nx.broadcast(1.0, {batch_size, seq_len, 1}) |> Nx.as_type({:f, 32})

        if cache == nil do
          {logits, cache_tuple} = jit_with_cache.(batch, batch_aux, embed_mask, full_params)
          cache_list = cache_tuple_to_list(cache_tuple)
          {logits, cache_list}
        else
          last_tokens = Enum.map(list_of_token_lists, fn seq -> [List.last(seq)] end)
          batch_one = Nx.tensor(last_tokens, type: {:s, 32})
          aux_one = Nx.broadcast(0.0, {batch_size, 1, 192}) |> Nx.as_type({:f, 32})
          mask_one = Nx.broadcast(1.0, {batch_size, 1, 1}) |> Nx.as_type({:f, 32})
          cache_to_use = maybe_replicate_cache(cache, batch_size)
          cache_tuple_to_use = cache_list_to_tuple(cache_to_use)

          {logits, new_cache_tuple} =
            jit_incremental.(batch_one, aux_one, mask_one, full_params, cache_tuple_to_use)

          new_cache = cache_tuple_to_list(new_cache_tuple)
          {logits, new_cache}
        end
      end

      {:ok, batch_fn}
    rescue
      e ->
        {:error, "Torchx JIT or defn params failed: #{inspect(e)}"}
    end
  end

  defp cache_tuple_to_list(cache_tuple) do
    Tuple.to_list(cache_tuple)
  end

  defp cache_list_to_tuple(cache_list) do
    List.to_tuple(cache_list)
  end

  defp replicate_cache(cache, batch_size) do
    Enum.map(cache, fn {k, v} ->
      {b, n_head, len, hd} = Nx.shape(k)

      if b == 1 do
        k_exp = Nx.broadcast(k, {batch_size, n_head, len, hd})
        v_exp = Nx.broadcast(v, {batch_size, n_head, len, hd})
        {k_exp, v_exp}
      else
        {k, v}
      end
    end)
  end

  defp maybe_replicate_cache(cache, batch_size) do
    case cache do
      [] ->
        []

      [{k, _} | _] ->
        b = elem(Nx.shape(k), 0)
        if b == 1 and batch_size > 1, do: replicate_cache(cache, batch_size), else: cache
    end
  end

  # Fallback when state has no get_logits_batch_fn (e.g. stub): one forward per sequence, then stack. Same API, no KV-cache.
  defp build_fallback_batch_fn(get_logits_fn) do
    fn list_of_token_lists, _cache ->
      logits =
        list_of_token_lists
        |> Enum.map(fn seq -> get_logits_fn.(seq) |> Nx.squeeze(axes: [0]) end)
        |> Nx.stack(axis: 0)

      {logits, nil}
    end
  end

  @doc """
  Convert item_ids (catalog indices) to left-padded token sequence for inference (same as Python serve seq_to_batch).
  Uses state.token_id_list when present, else state.token_id_map (when loaded from DB).
  """
  @spec item_ids_to_context_token_ids(
          [non_neg_integer()],
          [[non_neg_integer()]] | state(),
          non_neg_integer() | nil
        ) :: [integer()]
  def item_ids_to_context_token_ids(item_ids, token_id_list, padding_id \\ @padding_id)
      when is_list(token_id_list) or is_struct(token_id_list) do
    seq = Enum.take(item_ids, -@max_length)

    token_list =
      if is_struct(token_id_list) do
        state = token_id_list

        Enum.flat_map(seq, fn iid ->
          if state.token_id_map && state.token_id_map != %{} do
            Map.get(state.token_id_map, iid, [0, 0, 0, 0])
          else
            Enum.at(state.token_id_list || [], iid) || [0, 0, 0, 0]
          end
        end)
      else
        Enum.flat_map(seq, fn iid -> Enum.at(token_id_list, iid) || [0, 0, 0, 0] end)
      end

    len = length(token_list)
    padding = List.duplicate(padding_id || @padding_id, @seq_token_capacity - len)
    padding ++ token_list
  end

  @doc """
  Recommend next item(s) given context item_ids. Returns up to `top_k` item_ids (best first) from beam search.
  """
  @spec recommend(state(), [non_neg_integer()], pos_integer()) ::
          {:ok, [non_neg_integer()]} | {:error, String.t()}
  def recommend(state, item_ids, top_k \\ 5)
      when is_list(item_ids) and is_integer(top_k) and top_k >= 1 do
    if item_ids == [] do
      {:error, "item_ids must be non-empty"}
    else
      top_k = min(top_k, 20)
      context_token_ids = item_ids_to_context_token_ids(item_ids, state)
      batch_fn = state.get_logits_batch_fn || build_fallback_batch_fn(state.get_logits_fn)

      case Decode.beam_search_top_k(
             state.get_logits_fn,
             state.trie,
             context_token_ids,
             top_k,
             batch_fn
           ) do
        {:ok, list} -> {:ok, list}
        :not_found -> {:ok, []}
      end
    end
  end
end
