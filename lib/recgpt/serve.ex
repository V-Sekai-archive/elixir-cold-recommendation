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
  alias RecGPT.FuxiLinearInferenceDefn
  alias RecGPT.FuxiLinearInferenceParams
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
    :decode_constants,
    :embedding_cache,
    :token_cache
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
          get_logits_4_fn: (Nx.Tensor.t() -> Nx.Tensor.t()),
          inference_backend: term() | nil,
          beam_width_override: non_neg_integer() | nil,
          decode_constants:
            %{root_state: Nx.Tensor.t(), neg_inf: Nx.Tensor.t(), vocab_t: Nx.Tensor.t()} | nil,
          embedding_cache: :ets.table() | nil,
          token_cache: :ets.table() | nil
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
         {:ok, item_text} <- load_catalog(catalog_path, num_items) do
      trie = Trie.build(token_id_list)
      trie_tensors = Trie.to_tensors(trie, @vocab_size)
      trie_tensors = transfer_trie_tensors(trie_tensors, inference_backend)

      item_id_to_tokens_tensor =
        token_id_list
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
        token_id_map: nil,
        item_id_to_tokens_tensor: item_id_to_tokens_tensor,
        item_text: item_text,
        num_items: num_items,
        get_logits_4_fn: get_logits_4_fn,
        inference_backend: inference_backend,
        beam_width_override: beam_width_override,
        decode_constants: decode_constants
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
         {:ok, {embedding_cache, token_cache}} <- load_embedding_caches(inference_backend) do
      trie_tensors = Trie.to_tensors(trie, @vocab_size)
      trie_tensors = transfer_trie_tensors(trie_tensors, inference_backend)

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
        token_id_list: [],
        token_id_map: token_id_map,
        item_id_to_tokens_tensor: item_id_to_tokens_tensor,
        item_text: item_text,
        num_items: num_items,
        get_logits_4_fn: get_logits_4_fn,
        inference_backend: inference_backend,
        beam_width_override: beam_width_override,
        decode_constants: decode_constants,
        embedding_cache: embedding_cache,
        token_cache: token_cache
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

  defp load_embedding_caches(inference_backend) do
    alias RecGPT.EmbeddingCache

    case EmbeddingCache.load_from_db(inference_backend) do
      {:ok, tables} ->
        {:ok, tables}

      {:error, reason} ->
        # Warn but don't fail: embedding cache is optional
        IO.warn("Failed to load embedding cache: #{reason}")
        {:ok, nil, nil}
    end
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

  @doc """
  Build opts for Decode.beam_search_top_k_spmd (same as recommend uses).
  """
  def decode_opts(state, _context_ids) do
    [beam_width_override: state.beam_width_override, constants: state.decode_constants]
  end

  defp build_jit_single(fuxi?) do
    if fuxi? do
      Nx.Defn.jit(&FuxiLinearInferenceDefn.forward_last_4_logits/4, compiler: EXLA)
    else
      Nx.Defn.jit(&InferenceDefn.forward_last_4_logits/4, compiler: EXLA)
    end
  end

  defp build_get_logits_4_fn(params, inference_backend) do
    fuxi? = Inference.fuxi_checkpoint?(params)
    dtype = Application.get_env(:recgpt, :inference_dtype, {:bf, 16})

    defn_params =
      if fuxi? do
        FuxiLinearInferenceParams.build_defn_params(params, dtype)
      else
        n_layers = Inference.n_layers_from_params(params)
        InferenceParams.build_defn_params(params, n_layers, dtype)
      end

    defn_params = transfer_defn_params_to_backend(defn_params, inference_backend)
    jit_single = build_jit_single(fuxi?)
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

            # Cache up to 8 shapes; drop oldest by clearing when over limit
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

      decode_strategy = Application.get_env(:recgpt, :decode_strategy, :beam_search)

      use_mtp = decode_strategy in [:mtp, :lookahead, :direct_score]
      needs_beam = not use_mtp
      has_beam = state.trie_tensors && state.item_id_to_tokens_tensor && state.get_logits_4_fn
      has_mtp = state.item_id_to_tokens_tensor && state.get_logits_4_fn

      cond do
        needs_beam and !has_beam ->
          {:error,
           "Beam decode required: trie_tensors, item_id_to_tokens_tensor, and get_logits_4_fn must be set"}

        use_mtp and !has_mtp ->
          {:error,
           "MTP decode required: item_id_to_tokens_tensor and get_logits_4_fn must be set"}

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
