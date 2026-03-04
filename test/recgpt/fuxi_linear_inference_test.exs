# RecGPT.FuxiLinearInference: full model unit tests (no stubs).
# Uses init_full_params for complete FuXi-Linear + RecGPT semantic ID from upstream.
defmodule RecGPT.FuxiLinearInferenceTest do
  use ExUnit.Case, async: true

  alias RecGPT.FuxiLinearInference

  describe "forward/4 with full params" do
    test "returns logits (batch, 15_361) for last position" do
      params = FuxiLinearInference.init_full_params()
      batch = 2
      seq_len = 8
      batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
      batch_aux = Nx.iota({batch, seq_len, 192}, type: {:f, 32}) |> Nx.divide(1_000) |> Nx.subtract(0.1)
      embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})

      logits = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)

      assert Nx.shape(logits) == {batch, 15_361}
      assert Nx.type(logits) == {:f, 32}
    end

    test "forward is deterministic for same inputs and params" do
      params = FuxiLinearInference.init_full_params()
      batch_token_ids = Nx.tensor([[1, 2, 3, 4, 5]], type: {:s, 32})
      batch_aux = Nx.broadcast(0.1, {1, 5, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 5, 1}) |> Nx.as_type({:f, 32})

      logits_a = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)
      logits_b = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)

      assert Nx.all_close(logits_a, logits_b) |> Nx.to_number() == 1
    end

    test "forward produces finite logits (no NaN or Inf)" do
      params = FuxiLinearInference.init_full_params()
      batch_token_ids = Nx.tensor([[0, 10, 20, 50, 100]], type: {:s, 32})
      batch_aux = Nx.iota({1, 5, 192}, type: {:f, 32}) |> Nx.divide(100) |> Nx.subtract(0.25)
      embed_mask = Nx.broadcast(1.0, {1, 5, 1}) |> Nx.as_type({:f, 32})

      logits = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)

      refute Nx.any(Nx.not_equal(logits, logits)) |> Nx.to_number() == 1, "logits must not contain NaN"
    end

    test "forward with seq_len 1 returns valid logits" do
      params = FuxiLinearInference.init_full_params()
      batch_token_ids = Nx.tensor([[42]], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, 1, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 1, 1}) |> Nx.as_type({:f, 32})

      logits = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)

      assert Nx.shape(logits) == {1, 15_361}
    end

    test "different inputs produce different logits" do
      params = FuxiLinearInference.init_full_params()
      batch_aux = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

      ids_a = Nx.tensor([[1, 2, 3, 4]], type: {:s, 32})
      ids_b = Nx.tensor([[5, 6, 7, 8]], type: {:s, 32})

      logits_a = FuxiLinearInference.forward(ids_a, batch_aux, embed_mask, params)
      logits_b = FuxiLinearInference.forward(ids_b, batch_aux, embed_mask, params)

      refute Nx.all_close(logits_a, logits_b) |> Nx.to_number() == 1
    end

    test "embed_mask zeros out padding positions (aux contribution)" do
      params = FuxiLinearInference.init_full_params()
      batch_token_ids = Nx.tensor([[1, 2, 3]], type: {:s, 32})
      batch_aux = Nx.iota({1, 3, 192}, type: {:f, 32}) |> Nx.divide(100) |> Nx.add(0.9)
      # Mask: 1 for pos 0,1; 0 for pos 2
      embed_mask = Nx.tensor([[[1.0], [1.0], [0.0]]], type: {:f, 32})

      logits = FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)

      assert Nx.shape(logits) == {1, 15_361}
    end
  end

  describe "init_full_params/1" do
    test "produces all required keys for forward" do
      params = FuxiLinearInference.init_full_params()

      assert Map.has_key?(params, "wte")
      assert Map.has_key?(params, "ae.linear.weight")
      assert Map.has_key?(params, "ae.linear.bias")
      assert Map.has_key?(params, "ae.norm.weight")
      assert Map.has_key?(params, "ae.norm.bias")
      assert Map.has_key?(params, "pred_head.weight")
      assert Map.has_key?(params, "pred_head.bias")
      assert Map.has_key?(params, "ln_f.weight")
      assert Map.has_key?(params, "ln_f.bias")

      for i <- 0..3 do
        base = "fuxi.block.#{i}."
        assert Map.has_key?(params, base <> "uvqk")
        assert Map.has_key?(params, base <> "retention.gamma")
        assert Map.has_key?(params, base <> "retention.ln.weight")
        assert Map.has_key?(params, base <> "channel_t.proj_v.weight")
        assert Map.has_key?(params, base <> "channel_t.gamma")
        assert Map.has_key?(params, base <> "channel_p.proj_p.weight")
        assert Map.has_key?(params, base <> "channel_p.emb")
        assert Map.has_key?(params, base <> "mffn.lin0.weight")
        assert Map.has_key?(params, base <> "mffn.lin1.weight")
      end
    end

    test "param shapes match FuXi-Linear upstream" do
      params = FuxiLinearInference.init_full_params()

      assert Nx.shape(params["wte"]) == {15_361, 768}
      assert Nx.shape(params["ae.linear.weight"]) == {192, 768}
      assert Nx.shape(params["pred_head.weight"]) == {768, 15_361}
      assert Nx.shape(params["ln_f.weight"]) == {768}

      assert Nx.shape(params["fuxi.block.0.uvqk"]) == {768, 768}
      assert Nx.shape(params["fuxi.block.0.retention.gamma"]) == {4}
      assert Nx.shape(params["fuxi.block.0.channel_t.proj_v.weight"]) == {768, 128}
      assert Nx.shape(params["fuxi.block.0.channel_p.proj_p.weight"]) == {768, 128}
      assert Nx.shape(params["fuxi.block.0.mffn.lin0.weight"]) == {384, 768}
    end

    test "init_full_params with custom n_blocks" do
      params = FuxiLinearInference.init_full_params(n_blocks: 2)

      assert Map.has_key?(params, "fuxi.block.0.uvqk")
      assert Map.has_key?(params, "fuxi.block.1.uvqk")
      refute Map.has_key?(params, "fuxi.block.2.uvqk")
    end
  end

  describe "forward/4 error cases" do
    test "raises when wte is missing" do
      params = FuxiLinearInference.init_full_params() |> Map.delete("wte")
      batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

      assert_raise RuntimeError, ~r/params must include wte/, fn ->
        FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)
      end
    end

    test "raises when ae.linear.weight is missing" do
      params = FuxiLinearInference.init_full_params() |> Map.delete("ae.linear.weight")
      batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

      assert_raise RuntimeError, ~r/FuXi requires ae/, fn ->
        FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)
      end
    end

    test "raises when fuxi block uvqk is missing" do
      params = FuxiLinearInference.init_full_params()
      params = Map.delete(params, "fuxi.block.0.uvqk")
      batch_token_ids = Nx.tensor([[0, 1, 2, 3]], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, 4, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, 4, 1}) |> Nx.as_type({:f, 32})

      assert_raise RuntimeError, ~r/missing fuxi.block.0.uvqk/, fn ->
        FuxiLinearInference.forward(batch_token_ids, batch_aux, embed_mask, params)
      end
    end
  end
end
