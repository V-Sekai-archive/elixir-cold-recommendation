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
  alias RecGPT.Trie

  @padding_id 15_360
  @max_length 255
  @seq_token_capacity 1024
  @vocab_size 15_361

  defstruct [
    :params,
    :trie,
    :trie_tensors,
    :token_id_list,
    :token_id_map,
    :item_id_to_tokens_tensor,
    :item_text,
    :num_items,
    :get_logits_fn,
    :get_logits_batch_fn,
    :get_logits_batch_tensor_fn,
    :inference_backend
  ]

  @type state :: %__MODULE__{
          params: map(),
          trie: map(),
          trie_tensors: %{next_state: Nx.Tensor.t(), item_at_leaf: Nx.Tensor.t(), num_states: non_neg_integer()} | nil,
          token_id_list: [[non_neg_integer()]],
          token_id_map: %{non_neg_integer() => [non_neg_integer()]} | nil,
          item_id_to_tokens_tensor: Nx.Tensor.t() | nil,
          item_text: %{non_neg_integer() => String.t() | map()},
          num_items: non_neg_integer(),
          get_logits_fn: (list(non_neg_integer()) -> Nx.Tensor.t()),
          get_logits_batch_fn: (([[non_neg_integer()]], term()) -> {Nx.Tensor.t(), term()}) | nil,
          get_logits_batch_tensor_fn: (Nx.Tensor.t(), term() -> {Nx.Tensor.t(), term()}) | nil,
          inference_backend: term() | nil
        }

  @doc """
  Load server state: checkpoint export, fixture (token_id_list), optional catalog JSON.
  Returns {:ok, state} or {:error, reason}.
  """
  @spec load_state(String.t(), String.t(), String.t() | nil) ::
          {:ok, state()} | {:error, String.t()}
  def load_state(fixture_path, ckpt_export_dir, catalog_path \\ nil) do
    with :ok <- ensure_exla(),
         {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {params, inference_backend} <- maybe_transfer_params_to_exla(params),
         {:ok, token_id_list, num_items} <- load_fixture(fixture_path),
         {:ok, item_text} <- load_catalog(catalog_path, num_items),
         {:ok, get_logits_batch_fn} <- build_get_logits_batch_fn(params, inference_backend) do
      trie = Trie.build(token_id_list)
      get_logits_fn = build_get_logits_fn_from_batch(get_logits_batch_fn)

      trie_tensors = Trie.to_tensors(trie, @vocab_size)
      trie_tensors = transfer_trie_tensors(trie_tensors, inference_backend)

      item_id_to_tokens_tensor =
        token_id_list
        |> Nx.tensor(type: {:s, 32})
        |> Nx.backend_transfer(inference_backend)

      get_logits_batch_tensor_fn = build_get_logits_batch_tensor_fn(params, inference_backend)

      state = %__MODULE__{
        params: params,
        trie: trie,
        trie_tensors: trie_tensors,
        token_id_list: token_id_list,
        token_id_map: nil,
        item_id_to_tokens_tensor: item_id_to_tokens_tensor,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn,
        get_logits_batch_fn: get_logits_batch_fn,
        get_logits_batch_tensor_fn: get_logits_batch_tensor_fn,
        inference_backend: inference_backend
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
    with :ok <- ensure_exla(),
         {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {params, inference_backend} <- maybe_transfer_params_to_exla(params),
         {:ok, trie, token_id_map, num_items} <- load_fixture_from_db(),
         {:ok, item_text} <- load_catalog(catalog_path, num_items),
         {:ok, get_logits_batch_fn} <- build_get_logits_batch_fn(params, inference_backend) do
      get_logits_fn = build_get_logits_fn_from_batch(get_logits_batch_fn)

      trie_tensors = Trie.to_tensors(trie, @vocab_size)
      trie_tensors = transfer_trie_tensors(trie_tensors, inference_backend)

      item_id_to_tokens_tensor =
        for i <- 0..(num_items - 1), do: Map.get(token_id_map, i, [0, 0, 0, 0])
        |> Nx.tensor(type: {:s, 32})
        |> Nx.backend_transfer(inference_backend)

      get_logits_batch_tensor_fn = build_get_logits_batch_tensor_fn(params, inference_backend)

      state = %__MODULE__{
        params: params,
        trie: trie,
        trie_tensors: trie_tensors,
        token_id_list: [],
        token_id_map: token_id_map,
        item_id_to_tokens_tensor: item_id_to_tokens_tensor,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn,
        get_logits_batch_fn: get_logits_batch_fn,
        get_logits_batch_tensor_fn: get_logits_batch_tensor_fn,
        inference_backend: inference_backend
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

  defp ensure_exla do
    if Code.ensure_loaded?(EXLA) do
      case Application.ensure_all_started(:exla) do
        {:ok, _} -> :ok
        {:error, {app, reason}} ->
          {:error, "EXLA required for inference. exla app failed to start: #{inspect(app)} - #{inspect(reason)}"}
      end
    else
      {:error,
       "EXLA required for inference. Add {:exla, \"~> 0.10\"} to deps and ensure it compiles."}
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

  # Load on BinaryBackend; transfer params to EXLA (client from config, e.g. :cuda or :host) for inference.
  defp maybe_transfer_params_to_exla(params) do
    client = Application.get_env(:exla, :default_client, :host)
    backend = {EXLA.Backend, client: client}
    params_exla = Map.new(params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
    {params_exla, backend}
  end

  defp load_fixture(path) do
    if File.regular?(path) do
      fixture = File.read!(path) |> Jason.decode!()

      token_id_list =
        (fixture["token_id_list"] || []) |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

      num_items = fixture["num_items"] || length(token_id_list)

      # Single-path trie: multiple items but only one unique first token -> beam never expands.
      cond do
        token_id_list == [] ->
          {:ok, token_id_list, num_items}

        length(token_id_list) > 1 ->
          first_tokens = token_id_list |> Enum.map(&List.first/1) |> Enum.uniq()
          if length(first_tokens) == 1 do
            {:error,
             "Fixture has single-path trie (all items share the same first token). " <>
               "Rebuild with VAE FSQ: mix recgpt.fetch_vae_ckpt then mix recgpt.build_fixture " <>
               "--items data/steam/items.json --out #{path} --ckpt <ckpt_dir>"}
          else
            {:ok, token_id_list, num_items}
          end

        true ->
          {:ok, token_id_list, num_items}
      end
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

  defp build_get_logits_batch_fn(params, inference_backend) do
    try do
      # Use non-defn Inference (no JIT / Nx.Defn) so EXLA is used as plain backend only.
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

        {batch, batch_aux, embed_mask} =
          maybe_transfer_to_inference_backend({batch, batch_aux, embed_mask}, inference_backend)

        if cache == nil do
          {logits, cache_list} =
            Inference.forward_with_cache(batch, batch_aux, embed_mask, params)
          {logits, cache_list}
        else
          last_tokens = Enum.map(list_of_token_lists, fn seq -> [List.last(seq)] end)
          batch_one = Nx.tensor(last_tokens, type: {:s, 32})
          aux_one = Nx.broadcast(0.0, {batch_size, 1, 192}) |> Nx.as_type({:f, 32})
          mask_one = Nx.broadcast(1.0, {batch_size, 1, 1}) |> Nx.as_type({:f, 32})

          {batch_one, aux_one, mask_one} =
            maybe_transfer_to_inference_backend(
              {batch_one, aux_one, mask_one},
              inference_backend
            )

          cache_to_use = maybe_replicate_cache(cache, batch_size)

          {logits, new_cache} =
            Inference.forward_incremental(
              batch_one,
              aux_one,
              mask_one,
              params,
              cache_to_use
            )

          {logits, new_cache}
        end
      end

      {:ok, batch_fn}
    rescue
      e ->
        {:error, "Inference (non-defn) failed: #{inspect(e)}"}
    end
  end

  defp maybe_transfer_to_inference_backend(tensors, nil), do: tensors

  defp maybe_transfer_to_inference_backend({a, b, c}, backend) do
    {
      Nx.backend_transfer(a, backend),
      Nx.backend_transfer(b, backend),
      Nx.backend_transfer(c, backend)
    }
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

  defp transfer_trie_tensors(%{next_state: ns, item_at_leaf: ial}, backend) do
    %{
      next_state: Nx.backend_transfer(ns, backend),
      item_at_leaf: Nx.backend_transfer(ial, backend),
      num_states: Nx.shape(ns) |> elem(0)
    }
  end

  defp build_get_logits_batch_tensor_fn(params, inference_backend) do
    fn batch_tensor, cache ->
      {batch_size, seq_len} = Nx.shape(batch_tensor)
      aux =
        Nx.broadcast(0.0, {batch_size, seq_len, 192})
        |> Nx.as_type({:f, 32})
        |> Nx.backend_transfer(inference_backend)
      mask =
        Nx.broadcast(1.0, {batch_size, seq_len, 1})
        |> Nx.as_type({:f, 32})
        |> Nx.backend_transfer(inference_backend)

      if cache == nil do
        Inference.forward_with_cache(batch_tensor, aux, mask, params)
      else
        last_tokens = batch_tensor |> Nx.slice_along_axis(seq_len - 1, 1, axis: 1)
        aux_one =
          Nx.broadcast(0.0, {batch_size, 1, 192})
          |> Nx.as_type({:f, 32})
          |> Nx.backend_transfer(inference_backend)
        mask_one =
          Nx.broadcast(1.0, {batch_size, 1, 1})
          |> Nx.as_type({:f, 32})
          |> Nx.backend_transfer(inference_backend)
        cache_to_use = maybe_replicate_cache(cache, batch_size)
        Inference.forward_incremental(last_tokens, aux_one, mask_one, params, cache_to_use)
      end
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
  Recommend next items for multiple contexts in one batched pass.
  `list_of_contexts` is a list of item_id lists (each non-empty). Returns a list of
  `{:ok, [item_id, ...]}` or `{:error, msg}` in the same order. Empty contexts get `{:error, "item_ids must be non-empty"}`.
  Uses batched beam search for better throughput than calling `recommend/3` repeatedly.
  """
  @spec recommend_batch(state(), [[non_neg_integer()]], pos_integer()) ::
          [{:ok, [non_neg_integer()]} | {:error, String.t()}]
  def recommend_batch(state, list_of_contexts, top_k \\ 5)
      when is_list(list_of_contexts) and is_integer(top_k) and top_k >= 1 do
    top_k = min(top_k, 20)

    if state.trie_tensors && state.item_id_to_tokens_tensor && state.get_logits_batch_tensor_fn do
      # SPMD path: one recommend (SPMD decode) per context
      Enum.map(list_of_contexts, fn ctx ->
        recommend(state, ctx, top_k)
      end)
    else
      batch_fn = state.get_logits_batch_fn || build_fallback_batch_fn(state.get_logits_fn)
      non_empty =
        list_of_contexts
        |> Enum.with_index()
        |> Enum.filter(fn {ctx, _} -> ctx != [] end)

      if non_empty == [] do
        Enum.map(list_of_contexts, fn _ -> {:error, "item_ids must be non-empty"} end)
      else
        context_token_ids =
          Enum.map(non_empty, fn {ctx, _} -> item_ids_to_context_token_ids(ctx, state) end)

        results =
          Decode.beam_search_top_k_batched(state.trie, context_token_ids, top_k, batch_fn)

        idx_to_result =
          non_empty
          |> Enum.zip(results)
          |> Map.new(fn {{_, idx}, r} -> {idx, r} end)

        Enum.map(0..(length(list_of_contexts) - 1), fn i ->
          if Enum.at(list_of_contexts, i) == [] do
            {:error, "item_ids must be non-empty"}
          else
            case Map.get(idx_to_result, i, :not_found) do
              {:ok, list} -> {:ok, list}
              :not_found -> {:ok, []}
            end
          end
        end)
      end
    end
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

      result =
        Decode.beam_search_top_k(
          state.get_logits_fn,
          state.trie,
          context_token_ids,
          top_k,
          batch_fn
        )

      case result do
        {:ok, list} -> {:ok, list}
        :not_found -> {:ok, []}
      end
    end
  end
end
