# RecGPT.Inference: forward pass with dummy or loaded params.
defmodule RecGPT.InferenceTest do
  use ExUnit.Case, async: true

  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.InferenceDefn
  alias RecGPT.InferenceParams
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
  test "forward_with_cache + forward_incremental matches forward on last position (attention math)" do
    # One-layer GPT-2 params so attention runs
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

    # Full sequence of 3 tokens; we want logits at last position
    batch_token_ids = Nx.tensor([[10, 20, 30]], type: {:s, 32})
    batch_aux = Nx.broadcast(0.0, {1, 3, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 3, 1}) |> Nx.as_type({:f, 32})

    logits_full = Inference.forward(batch_token_ids, batch_aux, embed_mask, params)

    # Build same logits via cache: prefix [10, 20] then incremental step with token 30
    prefix_ids = Nx.tensor([[10, 20]], type: {:s, 32})
    prefix_aux = Nx.broadcast(0.0, {1, 2, 192}) |> Nx.as_type({:f, 32})
    prefix_mask = Nx.broadcast(1.0, {1, 2, 1}) |> Nx.as_type({:f, 32})

    {_logits_prefix, cache} =
      Inference.forward_with_cache(prefix_ids, prefix_aux, prefix_mask, params)

    last_token = Nx.tensor([[30]], type: {:s, 32})
    last_aux = Nx.broadcast(0.0, {1, 1, 192}) |> Nx.as_type({:f, 32})
    last_mask = Nx.broadcast(1.0, {1, 1, 1}) |> Nx.as_type({:f, 32})

    {logits_inc, _} =
      Inference.forward_incremental(last_token, last_aux, last_mask, params, cache)

    # Full forward last position must match incremental (same context + last token)
    assert Nx.shape(logits_full) == {1, 15_361}
    assert Nx.shape(logits_inc) == {1, 15_361}

    diff = Nx.subtract(logits_full, logits_inc) |> Nx.abs() |> Nx.reduce_max()

    assert Nx.to_number(diff) < 1.0e-2,
           "full forward last position should match incremental (diff max #{Nx.to_number(diff)})"
  end

  @tag :integration
  test "EXLA Defn forward_last_4_logits matches Inference.forward last position for stub params" do
    unless Code.ensure_loaded?(EXLA) do
      raise "EXLA not loaded; run with EXLA in deps to enable this test"
    end

    params = dummy_params()
    full_params = InferenceParams.build_defn_params(params, 0)
    jit_fn = Nx.Defn.jit(&InferenceDefn.forward_last_4_logits/4, compiler: Nx.Defn.Evaluator)

    # Need at least 4 tokens for forward_last_4_logits
    batch_token_ids = Nx.tensor([[10, 20, 30, 40]], type: {:s, 32})
    batch_aux = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

    logits_inference = Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    logits_4 = jit_fn.(batch_token_ids, batch_aux, embed_mask, full_params)
    assert Nx.shape(logits_4) == {1, 4, 15_361}

    # Last position logits from Defn (index 3) vs Inference full forward (last position)
    logits_defn_last = logits_4 |> Nx.slice_along_axis(3, 1, axis: 1) |> Nx.squeeze(axes: [1])
    logits_inference_last = logits_inference

    diff = Nx.subtract(logits_inference_last, logits_defn_last) |> Nx.abs() |> Nx.reduce_max()
    # Stub params with identity layers can have slight numerical drift between Evaluator and Inference
    assert Nx.to_number(diff) < 1.0e-1,
           "Inference and Defn last-position logits should match (diff max #{Nx.to_number(diff)})"
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
  test "load checkpoint + trie + beam_search_top_k_spmd returns next item_id (requires data/recgpt_ckpt_export)" do
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
    token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
    trie = Trie.build(token_id_list)
    trie_tensors = Trie.to_tensors(trie, 15_361)
    item_id_to_tokens = Nx.tensor(token_id_list, type: {:s, 32})

    get_logits_4_fn = fn context_tokens ->
      {batch_size, seq_len} = Nx.shape(context_tokens)
      batch_aux = Nx.broadcast(0.0, {batch_size, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {batch_size, seq_len, 1}) |> Nx.as_type({:f, 32})
      logits = Inference.forward_full_sequence(context_tokens, batch_aux, embed_mask, params)
      Nx.slice_along_axis(logits, seq_len - 4, 4, axis: 1)
    end

    # Context [0] = first item; predict next (item 0 or 1)
    result =
      Decode.beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        [0],
        1,
        get_logits_4_fn,
        Nx.default_backend(),
        trie
      )

    assert result in [{:ok, [0]}, {:ok, [1]}]
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
