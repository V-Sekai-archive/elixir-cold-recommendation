defmodule RecGPT.TestSupport.FrozenHelpers do
  @moduledoc """
  Shared helpers for tests: stub Serve state and frozen layer inputs.
  Use LayerFreeze.record_from_state to get frozen inputs from a full-weights (or stub) run.
  """
  alias RecGPT.CheckpointExport
  alias RecGPT.Inference
  alias RecGPT.LayerFreeze
  alias RecGPT.Serve
  alias RecGPT.Trie

  @stub_token_id_list [[100, 200, 300, 400], [101, 201, 301, 401]]

  @doc "Returns a minimal Serve state for unit tests (stub params, 2 items)."
  def build_stub_state do
    Application.ensure_all_started(:nx)
    trie = Trie.build(@stub_token_id_list)
    params = build_dummy_params()

    get_logits_fn = fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    %Serve{
      params: params,
      trie: trie,
      token_id_list: @stub_token_id_list,
      item_text: %{},
      num_items: 2,
      get_logits_fn: get_logits_fn
    }
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
