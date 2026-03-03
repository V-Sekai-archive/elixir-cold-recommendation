defmodule RecGPT.TestSupport.FrozenHelpers do
  @moduledoc """
  Shared helpers for tests: stub Serve state and frozen layer inputs.
  Use LayerFreeze.record_from_state to get frozen inputs from a full-weights (or stub) run.
  Stub state includes SPMD fields (trie_tensors, item_id_to_tokens_tensor, get_logits_4_fn)
  so Serve.recommend/3 works. Lives in lib/ so it compiles reliably for mix test.
  """
  alias RecGPT.CheckpointExport
  alias RecGPT.Inference
  alias RecGPT.InferenceDefn
  alias RecGPT.InferenceParams
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
    backend = Nx.default_backend()
    params_bin = transfer_params_to_backend(params, backend)

    n_layers = Inference.n_layers_from_params(params_bin)
    defn_params = InferenceParams.build_defn_params(params_bin, n_layers, {:f, 32})
    defn_params = Map.new(defn_params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

    jit_single = Nx.Defn.jit(&InferenceDefn.forward_last_4_logits/4, compiler: EXLA)

    get_logits_4_fn = fn context_tokens ->
      context_tokens = Nx.backend_transfer(context_tokens, backend)
      {batch_size, seq_len} = Nx.shape(context_tokens)

      aux =
        Nx.broadcast(0.0, {batch_size, seq_len, 192})
        |> Nx.as_type({:f, 32})
        |> Nx.backend_transfer(backend)

      mask =
        Nx.broadcast(1.0, {batch_size, seq_len, 1})
        |> Nx.as_type({:f, 32})
        |> Nx.backend_transfer(backend)

      jit_single.(context_tokens, aux, mask, defn_params)
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

    vocab_size = @vocab_size
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)
    neg_inf = Nx.tensor(-1.0e9, type: {:f, 32}) |> Nx.backend_transfer(backend)
    vocab_t = Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    decode_constants = %{root_state: root_state, neg_inf: neg_inf, vocab_t: vocab_t}

    %Serve{
      params: params_bin,
      trie: trie,
      trie_tensors: trie_tensors,
      token_id_list: token_id_list,
      token_id_map: nil,
      item_id_to_tokens_tensor: item_id_to_tokens_tensor,
      item_text: %{},
      num_items: length(token_id_list),
      get_logits_4_fn: get_logits_4_fn,
      inference_backend: backend,
      beam_width_override: nil,
      decode_constants: decode_constants
    }
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
end
