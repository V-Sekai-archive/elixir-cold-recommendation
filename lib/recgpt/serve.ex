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
    :beam_search_fused_fn,
    :inference_backend,
    :beam_width_override,
    :decode_constants,
    :context_cache
  ]

  @type state :: %__MODULE__{
          params: map(),
          trie: map(),
          trie_tensors:
            %{
              next_state: Nx.Tensor.t(),
              item_at_leaf: Nx.Tensor.t(),
              num_states: non_neg_integer()
            }
            | nil,
          token_id_list: [[non_neg_integer()]],
          token_id_map: %{non_neg_integer() => [non_neg_integer()]} | nil,
          item_id_to_tokens_tensor: Nx.Tensor.t() | nil,
          item_text: %{non_neg_integer() => String.t() | map()},
          num_items: non_neg_integer(),
          get_logits_fn: (list(non_neg_integer()) -> Nx.Tensor.t()),
          get_logits_batch_fn: ([[non_neg_integer()]], term() -> {Nx.Tensor.t(), term()}) | nil,
          get_logits_batch_tensor_fn: (Nx.Tensor.t(), term() -> {Nx.Tensor.t(), term()}) | nil,
          beam_search_fused_fn: (Nx.Tensor.t(), pos_integer() -> {:ok, Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()} | :unavailable) | nil,
          inference_backend: term() | nil,
          beam_width_override: non_neg_integer() | nil,
          decode_constants: %{root_state: Nx.Tensor.t(), neg_inf: Nx.Tensor.t(), vocab_t: Nx.Tensor.t()} | nil,
          context_cache: %{optional(integer()) => {Nx.Tensor.t(), term()}} | nil
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

      get_logits_batch_tensor_fn =
        build_get_logits_batch_tensor_fn(params, inference_backend, ckpt_export_dir)

      {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
      decode_constants = build_decode_constants(inference_backend, vocab_size)
      beam_width_override = Application.get_env(:recgpt, :beam_width_override)

      beam_search_fused_fn =
        build_beam_search_fused_fn(
          params,
          inference_backend,
          trie_tensors,
          decode_constants
        )

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
        beam_search_fused_fn: beam_search_fused_fn,
        inference_backend: inference_backend,
        beam_width_override: beam_width_override,
        decode_constants: decode_constants,
        context_cache: %{}
      }

      state = maybe_warm_context_cache(state)
      state = warm_fused_jit(state)
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
        for i <- 0..(num_items - 1),
            do:
              Map.get(token_id_map, i, [0, 0, 0, 0])
              |> Nx.tensor(type: {:s, 32})
              |> Nx.backend_transfer(inference_backend)

      get_logits_batch_tensor_fn =
        build_get_logits_batch_tensor_fn(params, inference_backend, ckpt_export_dir)

      {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
      decode_constants = build_decode_constants(inference_backend, vocab_size)
      beam_width_override = Application.get_env(:recgpt, :beam_width_override)

      beam_search_fused_fn =
        build_beam_search_fused_fn(
          params,
          inference_backend,
          trie_tensors,
          decode_constants
        )

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
        beam_search_fused_fn: beam_search_fused_fn,
        inference_backend: inference_backend,
        beam_width_override: beam_width_override,
        decode_constants: decode_constants,
        context_cache: %{}
      }

      # Context cache warming only supported when item_id_to_tokens_tensor is a single tensor (fixture path)
      state = maybe_warm_context_cache(state)
      state = warm_fused_jit(state)
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
        {:ok, _} ->
          :ok

        {:error, {app, reason}} ->
          {:error,
           "EXLA required for inference. exla app failed to start: #{inspect(app)} - #{inspect(reason)}"}
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
    # List-based batch fn (non-JIT); SPMD uses get_logits_batch_tensor_fn (JIT).
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
  end

  defp maybe_transfer_to_inference_backend({a, b, c}, backend) do
    {
      Nx.backend_transfer(a, backend),
      Nx.backend_transfer(b, backend),
      Nx.backend_transfer(c, backend)
    }
  end

  defp replicate_cache(cache, batch_size) do
    cache_list = if is_tuple(cache), do: Tuple.to_list(cache), else: cache

    Enum.map(cache_list, fn {k, v} ->
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

      cache when is_tuple(cache) or is_list(cache) ->
        first = if is_tuple(cache), do: elem(cache, 0), else: hd(cache)
        {k, _} = first
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

  defp build_decode_constants(backend, vocab_size) do
    dtype = Application.get_env(:recgpt, :inference_dtype, {:f, 32})
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)
    neg_inf = neg_inf_for_dtype(dtype) |> Nx.backend_transfer(backend)
    vocab_t = Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    %{root_state: root_state, neg_inf: neg_inf, vocab_t: vocab_t}
  end

  # FP8 E4M3 has limited range (~[-448, 448]); use -448 for mask.
  defp neg_inf_for_dtype(:f8_e4m3fn), do: Nx.tensor(-448.0, type: :f8_e4m3fn)
  defp neg_inf_for_dtype(dtype), do: Nx.tensor(-1.0e9, type: dtype)

  defp context_cache_key(context_ids) when is_list(context_ids) do
    :erlang.phash2(:erlang.term_to_binary(context_ids))
  end

  defp get_cached_step0(state, context_ids) do
    if state.context_cache && state.context_cache != %{} do
      key = context_cache_key(context_ids)
      Map.get(state.context_cache, key)
    else
      nil
    end
  end

  defp maybe_add_initial_step0(opts, state, context_ids) do
    case get_cached_step0(state, context_ids) do
      {logits, cache} -> Keyword.put(opts, :initial_step0, {logits, cache})
      nil -> opts
    end
  end

  defp maybe_add_fused_fn(opts, state) do
    if state.beam_search_fused_fn, do: Keyword.put(opts, :fused_fn, state.beam_search_fused_fn), else: opts
  end

  @doc """
  Build opts for Decode.beam_search_top_k_spmd (same as recommend uses).
  Used by trace_predict so the fused path and context cache are exercised.
  """
  def decode_opts(state, context_ids) do
    [beam_width_override: state.beam_width_override, constants: state.decode_constants]
    |> maybe_add_initial_step0(state, context_ids)
    |> maybe_add_fused_fn(state)
  end

  defp build_context_tokens_single(state, context_ids, backend) do
    item_id_to_tokens = state.item_id_to_tokens_tensor
    if is_list(item_id_to_tokens), do: nil
    if context_ids == [] do
      Nx.tensor([[0]], type: {:s, 32}) |> Nx.backend_transfer(backend)
    else
      context_ids_t = Nx.tensor(context_ids, type: {:s, 32}) |> Nx.backend_transfer(backend)
      context_ids_t = Nx.new_axis(context_ids_t, -1)
      ctx = Nx.gather(item_id_to_tokens, context_ids_t, axes: [0]) |> Nx.reshape({:auto})
      len = Nx.size(ctx)
      Nx.reshape(ctx, {1, len})
    end
  end

  defp build_context_tokens_batch(state, list_of_contexts, max_len, backend) do
    tensors =
      Enum.map(list_of_contexts, fn context_ids ->
        t = build_context_tokens_single(state, context_ids, backend)
        if t do
          {_b, len} = Nx.shape(t)
          pad_left = max_len - len
          if pad_left <= 0 do
            Nx.slice_along_axis(t, 0, max_len, axis: 1)
          else
            pad = Nx.broadcast(Nx.tensor(0, type: {:s, 32}), {1, pad_left}) |> Nx.backend_transfer(backend)
            Nx.concatenate([pad, t], axis: 1)
          end
        else
          nil
        end
      end)

    if Enum.any?(tensors, &is_nil/1), do: nil, else: Nx.concatenate(tensors, axis: 0)
  end

  defp slice_cache_at_index(cache_tuple, index) do
    list = Tuple.to_list(cache_tuple)
    sliced =
      Enum.map(list, fn {k, v} ->
        k1 = Nx.slice_along_axis(k, index, 1, axis: 0)
        v1 = Nx.slice_along_axis(v, index, 1, axis: 0)
        {k1, v1}
      end)
    List.to_tuple(sliced)
  end

  defp warm_fused_jit(state) do
    if Application.get_env(:recgpt, :warm_fused_jit, true) and
         not Application.get_env(:recgpt, :force_unfused_beam, true) do
      minimal = Nx.tensor([[0]], type: {:s, 32}) |> Nx.backend_transfer(state.inference_backend)
      _ = state.beam_search_fused_fn.(minimal, 5)
    end

    state
  end

  defp maybe_warm_context_cache(state) do
    warm_list = Application.get_env(:recgpt, :context_cache_warm_list, [])
    batch_size = Application.get_env(:recgpt, :context_cache_warm_batch_size, 4)

    if warm_list == [] or not is_list(warm_list) or is_list(state.item_id_to_tokens_tensor) do
      state
    else
      backend = state.inference_backend
      get_fn = state.get_logits_batch_tensor_fn
      max_len = Application.get_env(:recgpt, :max_cache_len, 128)

      cache =
        warm_list
        |> Enum.chunk_every(batch_size)
        |> Enum.reduce(state.context_cache, fn chunk, acc ->
          case build_context_tokens_batch(state, chunk, max_len, backend) do
            nil -> acc
            batch_tensor ->
              {logits_batch, cache_batch} = get_fn.(batch_tensor, nil)
              # logits_batch {b, seq_len, vocab}; we need last position per batch -> {b, vocab}
              seq_len = elem(Nx.shape(logits_batch), 1)
              logits_last = Nx.slice_along_axis(logits_batch, seq_len - 1, 1, axis: 1)
              logits_per = Nx.squeeze(logits_last, axes: [1])

              Enum.with_index(chunk)
              |> Enum.reduce(acc, fn {context_ids, i}, a ->
                logits_i = logits_per |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.squeeze(axes: [0])
                cache_i = slice_cache_at_index(cache_batch, i)
                key = context_cache_key(context_ids)
                Map.put(a, key, {logits_i, cache_i})
              end)
          end
        end)

      %{state | context_cache: cache}
    end
  end

  defp build_jit do
    jit_full = Nx.Defn.jit(&InferenceDefn.forward_with_cache/4, compiler: EXLA)
    jit_incr = Nx.Defn.jit(&InferenceDefn.forward_incremental/6, compiler: EXLA)
    {jit_full, jit_incr}
  end

  @pre_alloc_max_context 256
  @pre_alloc_max_beam 20

  defp build_beam_search_fused_fn(params, inference_backend, trie_tensors, decode_constants) do
    if Application.get_env(:recgpt, :force_unfused_beam, true) do
      fn _context_tokens, _request_beam_width -> :unavailable end
    else
      build_beam_search_fused_jit(params, inference_backend, trie_tensors, decode_constants)
    end
  end

  defp build_beam_search_fused_jit(params, inference_backend, trie_tensors, decode_constants) do
    n_layers = Inference.n_layers_from_params(params)
    dtype = Application.get_env(:recgpt, :inference_dtype, {:f, 32})
    defn_params = InferenceParams.build_defn_params(params, n_layers, dtype)
    defn_params = transfer_defn_params_to_backend(defn_params, inference_backend)

    fused_beam_width = Application.get_env(:recgpt, :fused_beam_width, 20) |> min(20) |> max(4)
    jit_fused = Nx.Defn.jit(InferenceDefn.beam_search_fused_fun_for_k(fused_beam_width), compiler: EXLA)

    pre_aux_full =
      Nx.broadcast(0.0, {1, @pre_alloc_max_context, 192})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)

    pre_mask_full =
      Nx.broadcast(1.0, {1, @pre_alloc_max_context, 1})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)

    past_len_0 = Nx.tensor(0, type: {:s, 32}) |> Nx.backend_transfer(inference_backend)
    past_len_1 = Nx.tensor(1, type: {:s, 32}) |> Nx.backend_transfer(inference_backend)
    past_len_2 = Nx.tensor(2, type: {:s, 32}) |> Nx.backend_transfer(inference_backend)

    next_state = trie_tensors.next_state
    item_at_leaf = trie_tensors.item_at_leaf
    root_state = decode_constants.root_state
    neg_inf = decode_constants.neg_inf
    vocab_t = decode_constants.vocab_t

    fn context_tokens, _request_beam_width ->
      # Skip fused if any input is a Defn expression (EXLA would raise); use unfused path instead.
      if not concrete_tensor?(context_tokens) do
        :unavailable
      else
        # Always run fused at compiled beam width; Decode slices to request_beam_width
        {_batch, context_len} = Nx.shape(context_tokens)
        context_len_scalar =
          Nx.tensor(context_len, type: {:s, 32}) |> Nx.backend_transfer(inference_backend)

        aux_0 = pre_aux_full |> Nx.slice_along_axis(0, context_len, axis: 1)
        mask_0 = pre_mask_full |> Nx.slice_along_axis(0, context_len, axis: 1)

        RecGPT.NVTX.range_push("beam_search_fused")
        {item_ids, beam_scores, prefix_tokens} =
          jit_fused.(
            context_tokens,
            context_len_scalar,
            past_len_0,
            past_len_1,
            past_len_2,
            aux_0,
            mask_0,
            defn_params,
            next_state,
            item_at_leaf,
            root_state,
            neg_inf,
            vocab_t
          )
        RecGPT.NVTX.range_pop()

        {:ok, item_ids, beam_scores, prefix_tokens}
      end
    end
  end

  defp concrete_tensor?(%Nx.Tensor{data: data}), do: not is_struct(data, Nx.Defn.Expr)
  defp concrete_tensor?(_), do: false

  defp build_get_logits_batch_tensor_fn(params, inference_backend, _ckpt_export_dir) do
    n_layers = Inference.n_layers_from_params(params)
    dtype = Application.get_env(:recgpt, :inference_dtype, {:f, 32})
    defn_params = InferenceParams.build_defn_params(params, n_layers, dtype)
    defn_params = transfer_defn_params_to_backend(defn_params, inference_backend)

    {jit_full, jit_incr} = build_jit()

    # Pre-alloc aux/mask on device for common shapes to avoid repeated alloc+transfer per request
    pre_aux_full =
      Nx.broadcast(0.0, {1, @pre_alloc_max_context, 192})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)
    pre_mask_full =
      Nx.broadcast(1.0, {1, @pre_alloc_max_context, 1})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)

    pre_aux_incr =
      Nx.broadcast(0.0, {@pre_alloc_max_beam, 1, 192})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)
    pre_mask_incr =
      Nx.broadcast(1.0, {@pre_alloc_max_beam, 1, 1})
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(inference_backend)

    fn batch_tensor, cache ->
      {batch_size, seq_len} = Nx.shape(batch_tensor)

      if cache == nil do
        RecGPT.NVTX.range_push("forward_with_cache")
        {aux, mask} =
          if seq_len <= @pre_alloc_max_context and batch_size == 1 do
            {
              pre_aux_full |> Nx.slice_along_axis(0, seq_len, axis: 1),
              pre_mask_full |> Nx.slice_along_axis(0, seq_len, axis: 1)
            }
          else
            aux =
              Nx.broadcast(0.0, {batch_size, seq_len, 192})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)
            mask =
              Nx.broadcast(1.0, {batch_size, seq_len, 1})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)
            {aux, mask}
          end
        {logits, cache_tuple} = jit_full.(batch_tensor, aux, mask, defn_params)
        RecGPT.NVTX.range_pop()
        padded = pad_cache_to_fixed(cache_tuple, inference_backend)
        {logits, padded}
      else
        last_tokens = batch_tensor |> Nx.slice_along_axis(seq_len - 1, 1, axis: 1)
        cache_to_use = maybe_replicate_cache(cache, batch_size)
        cache_tuple = ensure_cache_tuple(cache_to_use)
        past_len =
          Nx.tensor(seq_len - 1, type: {:s, 32}) |> Nx.backend_transfer(inference_backend)

        RecGPT.NVTX.range_push("forward_incremental")
        {aux_one, mask_one} =
          if batch_size <= @pre_alloc_max_beam do
            {
              pre_aux_incr |> Nx.slice_along_axis(0, batch_size, axis: 0),
              pre_mask_incr |> Nx.slice_along_axis(0, batch_size, axis: 0)
            }
          else
            aux_one =
              Nx.broadcast(0.0, {batch_size, 1, 192})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)
            mask_one =
              Nx.broadcast(1.0, {batch_size, 1, 1})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)
            {aux_one, mask_one}
          end
        {logits, new_cache} =
          jit_incr.(last_tokens, aux_one, mask_one, defn_params, cache_tuple, past_len)
        RecGPT.NVTX.range_pop()
        {logits, new_cache}
      end
    end
  end

  defp transfer_defn_params_to_backend(defn_params, backend) do
    Map.new(defn_params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
  end

  defp ensure_cache_tuple(cache) when is_list(cache), do: List.to_tuple(cache)
  defp ensure_cache_tuple(cache) when is_tuple(cache), do: cache

  defp pad_cache_to_fixed(cache_tuple, backend) do
    max_len = Application.get_env(:recgpt, :max_cache_len, 128)
    list = Tuple.to_list(cache_tuple)

    padded =
      Enum.map(list, fn {k, v} ->
        {b, n_head, len, hd} = Nx.shape(k)
        pad_count = max(0, max_len - len)
        ttype = Nx.type(k)

        if pad_count == 0 do
          {k, v}
        else
          # Create zero scalar on target backend to avoid host→device transfer of padding
          zero = Nx.tensor(0, type: ttype) |> Nx.backend_transfer(backend)
          zeros_k = Nx.broadcast(zero, {b, n_head, pad_count, hd})
          zeros_v = Nx.broadcast(zero, {b, n_head, pad_count, hd})

          {Nx.concatenate([k, zeros_k], axis: 2) |> Nx.backend_transfer(backend),
           Nx.concatenate([v, zeros_v], axis: 2) |> Nx.backend_transfer(backend)}
        end
      end)

    List.to_tuple(padded)
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

    Enum.map(list_of_contexts, fn ctx ->
      recommend(state, ctx, top_k)
    end)
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

      if !state.trie_tensors || !state.item_id_to_tokens_tensor || !state.get_logits_batch_tensor_fn do
        {:error,
         "SPMD decode required: trie_tensors, item_id_to_tokens_tensor, and get_logits_batch_tensor_fn must be set (both load_state and load_state_from_db provide these)"}
      else
        opts = decode_opts(state, item_ids)

        result =
          Decode.beam_search_top_k_spmd(
            state.trie_tensors,
            state.item_id_to_tokens_tensor,
            item_ids,
            top_k,
            state.get_logits_batch_tensor_fn,
            state.inference_backend,
            state.trie,
            opts
          )

        case result do
          {:ok, list} -> {:ok, list}
          :not_found -> {:ok, []}
        end
      end
    end
  end
end
