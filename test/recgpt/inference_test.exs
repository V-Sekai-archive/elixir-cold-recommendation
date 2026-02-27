# RecGPT.Inference: forward pass with dummy or loaded params.
defmodule RecGPT.InferenceTest do
  use ExUnit.Case, async: true

  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.Trie

  defp dummy_params do
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})

    %{
      "wte" => wte,
      "pred_head.weight" => head_w,
      "pred_head.bias" => head_b
    }
  end

  test "forward returns logits (batch, 15_361) with dummy params" do
    params = dummy_params()
    batch = 2
    seq_len = 4
    batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {batch, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})

    logits = Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    assert Nx.shape(logits) == {batch, 15_361}
  end

  test "forward_full_sequence returns logits (batch, seq_len, 15_361) for training" do
    params = dummy_params()
    batch = 2
    seq_len = 8
    batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {batch, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})

    logits =
      Inference.forward_full_sequence(batch_token_ids, batch_aux_embeds, embed_mask, params)

    assert Nx.shape(logits) == {batch, seq_len, 15_361}
  end

  test "forward with single batch and no aux params (no ae keys)" do
    params = dummy_params()
    batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    logits = Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    assert Nx.shape(logits) == {1, 15_361}
  end

  test "forward with seq_len 1 returns logits for single position" do
    params = dummy_params()
    batch_token_ids = Nx.tensor([[42]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 1, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 1, 1}) |> Nx.as_type({:f, 32})

    logits = Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    assert Nx.shape(logits) == {1, 15_361}
  end

  test "forward raises when wte is missing" do
    params = %{"pred_head.weight" => Nx.iota({15_361, 768})}
    batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    assert_raise RuntimeError, ~r/missing wte/, fn ->
      Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    end
  end

  test "forward raises when pred_head.weight is missing" do
    params = dummy_params() |> Map.delete("pred_head.weight") |> Map.delete("pred_head.bias")
    batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    assert_raise RuntimeError, ~r/missing pred_head/, fn ->
      Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    end
  end

  @tag :integration
  test "forward runs full GPT-2 backbone when gpt2model.h.0.* params present" do
    # Minimal one-layer GPT-2 params (small so test is fast)
    params =
      dummy_params()
      |> Map.put("gpt2model.h.0.ln_1.weight", Nx.broadcast(1.0, {768}) |> Nx.as_type({:f, 32}))
      |> Map.put("gpt2model.h.0.ln_1.bias", Nx.broadcast(0.0, {768}) |> Nx.as_type({:f, 32}))
      |> Map.put(
        "gpt2model.h.0.attn.c_attn.weight",
        Nx.iota({2304, 768}) |> Nx.divide(2304 * 768) |> Nx.as_type({:f, 32})
      )
      |> Map.put(
        "gpt2model.h.0.attn.c_attn.bias",
        Nx.broadcast(0.0, {2304}) |> Nx.as_type({:f, 32})
      )
      |> Map.put(
        "gpt2model.h.0.attn.c_proj.weight",
        Nx.iota({768, 768}) |> Nx.divide(768 * 768) |> Nx.as_type({:f, 32})
      )
      |> Map.put(
        "gpt2model.h.0.attn.c_proj.bias",
        Nx.broadcast(0.0, {768}) |> Nx.as_type({:f, 32})
      )
      |> Map.put("gpt2model.h.0.ln_2.weight", Nx.broadcast(1.0, {768}) |> Nx.as_type({:f, 32}))
      |> Map.put("gpt2model.h.0.ln_2.bias", Nx.broadcast(0.0, {768}) |> Nx.as_type({:f, 32}))
      |> Map.put(
        "gpt2model.h.0.mlp.c_fc.weight",
        Nx.iota({3072, 768}) |> Nx.divide(3072 * 768) |> Nx.as_type({:f, 32})
      )
      |> Map.put("gpt2model.h.0.mlp.c_fc.bias", Nx.broadcast(0.0, {3072}) |> Nx.as_type({:f, 32}))
      |> Map.put(
        "gpt2model.h.0.mlp.c_proj.weight",
        Nx.iota({768, 3072}) |> Nx.divide(768 * 3072) |> Nx.as_type({:f, 32})
      )
      |> Map.put(
        "gpt2model.h.0.mlp.c_proj.bias",
        Nx.broadcast(0.0, {768}) |> Nx.as_type({:f, 32})
      )
      |> Map.put("gpt2model.ln_f.weight", Nx.broadcast(1.0, {768}) |> Nx.as_type({:f, 32}))
      |> Map.put("gpt2model.ln_f.bias", Nx.broadcast(0.0, {768}) |> Nx.as_type({:f, 32}))

    batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    logits = Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    assert Nx.shape(logits) == {1, 15_361}
  end

  @tag :integration
  @tag :load_ckpt
  test "load real checkpoint export and run forward (requires data/recgpt_ckpt_export)" do
    export_dir = ckpt_export_dir()
    manifest_path = Path.join(export_dir, "manifest.json")

    unless File.regular?(manifest_path) do
      raise """
      Checkpoint export not found. From repo root run:
        python scripts/inspect_recgpt_checkpoint.py --export data/recgpt_ckpt_export
      Then run: mix test test/recgpt/inference_test.exs --include load_ckpt --include integration
      """
    end

    params = RecGPT.CheckpointLoader.load_from_export(export_dir)
    assert map_size(params) > 0
    assert params["gpt2model.wte.weight"] || params["gpt2model.wte"]

    # One sequence of 4 tokens (one item), no aux
    batch_token_ids = Nx.tensor([[100, 200, 300, 400]], type: {:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    logits = Inference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    assert Nx.shape(logits) == {1, 15_361}
    # Logits should be finite (no NaN; backend may not have is_finite)
    assert Nx.all(Nx.equal(logits, logits)) |> Nx.to_number() == 1
  end

  @tag :integration
  @tag :load_ckpt
  test "load checkpoint + trie + beam_search returns next item_id (requires data/recgpt_ckpt_export)" do
    export_dir = ckpt_export_dir()
    manifest_path = Path.join(export_dir, "manifest.json")

    unless File.regular?(manifest_path) do
      raise """
      Checkpoint export not found. From repo root run:
        python scripts/inspect_recgpt_checkpoint.py --export data/recgpt_ckpt_export
      Then run: mix test test/recgpt/inference_test.exs --include load_ckpt --include integration
      """
    end

    params = RecGPT.CheckpointLoader.load_from_export(export_dir)
    # Catalog: two items so beam search has two valid paths
    token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
    trie = Trie.build(token_id_list)

    get_logits_fn = fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    # Context = first item's tokens; predict next item (should resolve to item 0 or 1)
    context = [100, 200, 300, 400]
    result = Decode.beam_search(get_logits_fn, trie, context, 4)

    assert result in [{:ok, 0}, {:ok, 1}]
  end

  defp ckpt_export_dir do
    cwd = File.cwd!()

    candidates = [
      Path.expand("../data/recgpt_ckpt_export", cwd),
      Path.join(cwd, "data/recgpt_ckpt_export")
    ]

    Enum.find(candidates, fn p -> File.regular?(Path.join(p, "manifest.json")) end) ||
      Path.join(cwd, "data/recgpt_ckpt_export")
  end
end
