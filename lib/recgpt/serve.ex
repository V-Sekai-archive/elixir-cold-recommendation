defmodule RecGPT.Serve do
  @moduledoc """
  Next-item recommendation engine. Loads fixture + checkpoint, provides recommend/recommend_batch.
  """
  alias RecGPT.CheckpointLoader
  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.InferenceDefn
  alias RecGPT.InferenceParams
  alias RecGPT.Trie

  @padding_id 15_360
  @max_length 255
  @seq_token_capacity 2048
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
    :get_logits_4_fn,
    :inference_backend,
    :beam_width_override,
    :decode_constants
  ]

  @type state :: %__MODULE__{}

  @doc """
  Load state from fixture JSON + checkpoint export dir.
  """
  def load_state(fixture_path, ckpt_export_dir, _catalog_path \\ nil) do
    with :ok <- ensure_exla(),
         {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {params, inference_backend} <- maybe_transfer_params_to_exla(params),
         {:ok, token_id_list, num_items} <- load_fixture(fixture_path) do
      trie = Trie.build(token_id_list)
      trie_tensors = Trie.to_tensors(trie, @vocab_size)
      trie_tensors = transfer_trie_tensors(trie_tensors, inference_backend)

      token_id_map =
        token_id_list
        |> Enum.with_index()
        |> Map.new(fn {tokens, idx} -> {idx, tokens} end)

      item_id_to_tokens_tensor =
        for i <- 0..(num_items - 1),
            do:
              Map.get(token_id_map, i, [0, 0, 0, 0])
              |> Nx.tensor(type: {:s, 32})
              |> Nx.backend_transfer(inference_backend)

      get_logits_4_fn = build_get_logits_4_fn(params, inference_backend)
      {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
      decode_constants = build_decode_constants(inference_backend, vocab_size)
      beam_width_override = Application.get_env(:recgpt, :beam_width_override)

      state = %__MODULE__{
        params: params,
        trie: trie,
        trie_tensors: trie_tensors,
        token_id_list: token_id_list,
        token_id_map: token_id_map,
        item_id_to_tokens_tensor: item_id_to_tokens_tensor,
        item_text: %{},
        num_items: num_items,
        get_logits_4_fn: get_logits_4_fn,
        inference_backend: inference_backend,
        beam_width_override: beam_width_override,
        decode_constants: decode_constants
      }

      {:ok, state}
    end
  end

  defp ensure_exla do
    if Code.ensure_loaded?(EXLA) do
      case Application.ensure_all_started(:exla) do
        {:ok, _} -> :ok
        {:error, {app, reason}} ->
          {:error, "EXLA failed to start: #{inspect(app)} - #{inspect(reason)}"}
      end
    else
      {:error, "EXLA required. Add {:exla, \"~> 0.10\"} to deps."}
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

      cond do
        token_id_list == [] ->
          {:ok, token_id_list, num_items}

        length(token_id_list) > 1 ->
          first_tokens = token_id_list |> Enum.map(&List.first/1) |> Enum.uniq()

          if length(first_tokens) == 1 do
            {:error,
             "Fixture has single-path trie (all items share the same first token). " <>
               "Rebuild with VAE FSQ."}
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

  defp transfer_trie_tensors(%{next_state: ns, item_at_leaf: ial}, backend) do
    %{
      next_state: Nx.backend_transfer(ns, backend),
      item_at_leaf: Nx.backend_transfer(ial, backend),
      num_states: Nx.shape(ns) |> elem(0)
    }
  end

  defp build_decode_constants(backend, vocab_size) do
    dtype = Application.get_env(:recgpt, :inference_dtype, {:bf, 16})
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)
    neg_inf = neg_inf_for_dtype(dtype) |> Nx.backend_transfer(backend)
    vocab_t = Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    %{root_state: root_state, neg_inf: neg_inf, vocab_t: vocab_t}
  end

  defp neg_inf_for_dtype(dtype), do: Nx.tensor(-1.0e9, type: dtype)

  defp build_jit_single(_fuxi?) do
    Nx.Defn.jit(&InferenceDefn.forward_last_4_logits/4, compiler: EXLA)
  end

  defp build_get_logits_4_fn(params, inference_backend) do
    dtype = Application.get_env(:recgpt, :inference_dtype, {:bf, 16})
    _n_layers = Inference.n_layers_from_params(params)

    defn_params = InferenceParams.build_defn_params(params, dtype)
    defn_params = transfer_defn_params_to_backend(defn_params, inference_backend)
    jit_single = build_jit_single(false)
    cache_ref = :ets.new(:recgpt_aux_mask_cache, [:set, :private])

    fn context_tokens ->
      {batch_size, seq_len} = Nx.shape(context_tokens)
      shape = {batch_size, seq_len}
      dtype = Application.get_env(:recgpt, :inference_dtype, {:bf, 16})

      {aux, mask} =
        case :ets.lookup(cache_ref, shape) do
          [{^shape, a, m}] ->
            {a, m}

          [] ->
            a =
              Nx.broadcast(0.0, {batch_size, seq_len, 192})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)

            m =
              Nx.broadcast(1.0, {batch_size, seq_len, 1})
              |> Nx.as_type(dtype)
              |> Nx.backend_transfer(inference_backend)

            n = length(:ets.match_object(cache_ref, {:"$1", :_, :_}))
            if n >= 8, do: :ets.delete_all_objects(cache_ref)
            :ets.insert(cache_ref, {shape, a, m})
            {a, m}
        end

      jit_single.(context_tokens, aux, mask, defn_params)
    end
  end

  defp transfer_defn_params_to_backend(defn_params, backend) do
    Map.new(defn_params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
  end

  def item_ids_to_context_token_ids(item_ids, token_id_list, padding_id \\ @padding_id)
      when is_list(token_id_list) do
    seq = Enum.take(item_ids, -@max_length)

    token_list =
      Enum.flat_map(seq, fn iid -> Enum.at(token_id_list, iid) || [0, 0, 0, 0] end)

    len = length(token_list)
    padding = List.duplicate(padding_id || @padding_id, @seq_token_capacity - len)
    padding ++ token_list
  end

  def recommend_batch(state, list_of_contexts, top_k \\ 5)
      when is_list(list_of_contexts) and is_integer(top_k) and top_k >= 1 do
    top_k = min(top_k, 20)

    Enum.map(list_of_contexts, fn ctx ->
      recommend(state, ctx, top_k)
    end)
  end

  def recommend(state, item_ids, top_k \\ 5)
      when is_list(item_ids) and is_integer(top_k) and top_k >= 1 do
    if item_ids == [] do
      {:error, "item_ids must be non-empty"}
    else
      top_k = min(top_k, 20)

      decode_strategy = Application.get_env(:recgpt, :decode_strategy, :beam_search)

      use_mtp = decode_strategy in [:mtp, :lookahead, :direct_score]
      needs_beam = not use_mtp
      has_beam = state.trie_tensors && state.item_id_to_tokens_tensor && state.get_logits_4_fn
      has_mtp = state.item_id_to_tokens_tensor && state.get_logits_4_fn

      cond do
        needs_beam and !has_beam ->
          {:error, "Beam decode required: trie_tensors missing"}

        use_mtp and !has_mtp ->
          {:error, "MTP decode required: item_id_to_tokens_tensor missing"}

        use_mtp ->
          result =
            Decode.lookahead_top_k(
              state.item_id_to_tokens_tensor,
              item_ids,
              top_k,
              state.get_logits_4_fn,
              state.inference_backend
            )

          case result do
            {:ok, list} -> {:ok, list}
            :not_found -> {:ok, []}
          end

        true ->
          opts = [
            beam_width_override: state.beam_width_override,
            constants: state.decode_constants
          ]

          result =
            Decode.beam_search_top_k_spmd(
              state.trie_tensors,
              state.item_id_to_tokens_tensor,
              item_ids,
              top_k,
              state.get_logits_4_fn,
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
