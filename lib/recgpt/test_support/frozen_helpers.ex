defmodule RecGPT.TestSupport.FrozenHelpers do
  @moduledoc """
  Shared helpers for tests: stub Serve state and frozen layer inputs.
  Use LayerFreeze.record_from_state to get frozen inputs from a full-weights (or stub) run.
  Stub state includes SPMD fields (trie_tensors, item_id_to_tokens_tensor, get_logits_batch_tensor_fn)
  so Serve.recommend/3 works. Lives in lib/ so it compiles reliably for mix test.
  """
  alias RecGPT.CheckpointExport
  alias RecGPT.Inference
  alias RecGPT.LayerFreeze
  alias RecGPT.Serve
  alias RecGPT.Trie

  @vocab_size 15_361

  @doc "Returns a minimal Serve state for unit tests (stub params, 2 items)."
  def build_stub_state do
    build_stub_state(2)
  end

  @doc "Returns a Serve state with n items (stub params, token_id_list of length n)."
  def build_stub_state(n) when is_integer(n) and n >= 1 do
    token_id_list =
      Enum.map(0..(n - 1), fn i ->
        [100 + i, 200 + i, 300 + i, 400 + i]
      end)

    build_stub_state_with_token_id_list(token_id_list)
  end

  defp build_stub_state_with_token_id_list(token_id_list) do
    Application.ensure_all_started(:nx)
    trie = Trie.build(token_id_list)
    params = build_dummy_params()
    # Use default backend (EXLA when available) so trie + batch fn stay on same backend;
    # BinaryBackend would work but is very slow and can timeout.
    backend = Nx.default_backend()

    get_logits_fn = fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    {:ok, get_logits_batch_fn} = build_get_logits_batch_fn(params, backend)

    get_logits_batch_tensor_fn = fn batch_tensor, cache ->
      # Ensure batch_tensor and cache match our backend (batch fn uses backend)
      batch_tensor = Nx.backend_transfer(batch_tensor, backend)
      cache = if cache != nil, do: transfer_cache_to_backend(cache, backend), else: nil
      rows = Nx.to_list(batch_tensor)
      {logits, new_cache} = get_logits_batch_fn.(rows, cache)
      # Keep outputs on backend
      logits = Nx.backend_transfer(logits, backend)
      new_cache = if is_list(new_cache) and new_cache != [], do: transfer_cache_to_backend(new_cache, backend), else: new_cache
      {logits, new_cache}
    end

    trie_tensors = Trie.to_tensors(trie, @vocab_size)
    trie_tensors = %{
      next_state: Nx.backend_transfer(trie_tensors.next_state, backend),
      item_at_leaf: Nx.backend_transfer(trie_tensors.item_at_leaf, backend),
      num_states: Nx.shape(trie_tensors.next_state) |> elem(0)
    }

    item_id_to_tokens_tensor =
      token_id_list
      |> Nx.tensor(type: {:s, 32})
      |> Nx.backend_transfer(backend)

    %Serve{
      params: params,
      trie: trie,
      trie_tensors: trie_tensors,
      token_id_list: token_id_list,
      token_id_map: nil,
      item_id_to_tokens_tensor: item_id_to_tokens_tensor,
      item_text: %{},
      num_items: length(token_id_list),
      get_logits_fn: get_logits_fn,
      get_logits_batch_fn: get_logits_batch_fn,
      get_logits_batch_tensor_fn: get_logits_batch_tensor_fn,
      inference_backend: backend
    }
  end

  defp build_get_logits_batch_fn(params, backend) do
    # Keep all tensors on given backend to avoid EXLA/Binary mix in Decode
    params_bin = transfer_params_to_backend(params, backend)

    batch_fn = fn list_of_token_lists, cache when is_list(list_of_token_lists) and list_of_token_lists != [] ->
      max_len = list_of_token_lists |> Enum.map(&length/1) |> Enum.max()

      padded =
        Enum.map(list_of_token_lists, fn tokens ->
          len = length(tokens)
          padding = List.duplicate(15_360, max_len - len)
          padding ++ tokens
        end)

      batch = Nx.tensor(padded, type: {:s, 32}) |> Nx.backend_transfer(backend)
      {batch_size, seq_len} = Nx.shape(batch)
      batch_aux = Nx.broadcast(0.0, {batch_size, seq_len, 192}) |> Nx.as_type({:f, 32}) |> Nx.backend_transfer(backend)
      embed_mask = Nx.broadcast(1.0, {batch_size, seq_len, 1}) |> Nx.as_type({:f, 32}) |> Nx.backend_transfer(backend)

      if cache == nil do
        {logits, cache_list} = Inference.forward_with_cache(batch, batch_aux, embed_mask, params_bin)
        {logits, cache_list}
      else
        last_tokens = Enum.map(list_of_token_lists, fn seq -> [List.last(seq)] end)
        batch_one = Nx.tensor(last_tokens, type: {:s, 32}) |> Nx.backend_transfer(backend)
        aux_one = Nx.broadcast(0.0, {batch_size, 1, 192}) |> Nx.as_type({:f, 32}) |> Nx.backend_transfer(backend)
        mask_one = Nx.broadcast(1.0, {batch_size, 1, 1}) |> Nx.as_type({:f, 32}) |> Nx.backend_transfer(backend)

        {logits, new_cache} =
          Inference.forward_incremental(batch_one, aux_one, mask_one, params_bin, cache)

        {logits, new_cache}
      end
    end

    {:ok, batch_fn}
  end

  defp transfer_params_to_backend(params, backend) when is_map(params) do
    Map.new(params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
  end

  @doc "Frozen inputs from stub state for layer isolation (Recommendation/Model)."
  def build_frozen(context_item_ids \\ [0]) do
    state = build_stub_state()
    LayerFreeze.record_from_state(state, context_item_ids)
  end

  @doc "Writes a minimal checkpoint export to dir for load_state tests."
  def write_stub_ckpt!(dir) do
    File.mkdir_p!(dir)

    params = %{
      "wte" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.weight" =>
        Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    }

    CheckpointExport.write_export(params, dir)
  end

  defp build_dummy_params do
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    %{"wte" => wte, "pred_head.weight" => head_w, "pred_head.bias" => head_b}
  end

  defp transfer_cache_to_backend(cache, backend) when is_list(cache) do
    Enum.map(cache, fn {k, v} ->
      {Nx.backend_transfer(k, backend), Nx.backend_transfer(v, backend)}
    end)
  end

  defp transfer_cache_to_backend(cache, backend) when is_tuple(cache) do
    cache |> Tuple.to_list() |> transfer_cache_to_backend(backend) |> List.to_tuple()
  end
end
