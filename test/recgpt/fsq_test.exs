defmodule RecGPT.FSQTest do
  use ExUnit.Case, async: true

  alias RecGPT.FSQ

  describe "basis/0" do
    test "returns tensor of length 5, cumprod of level prefix" do
      b = FSQ.basis()
      assert Nx.size(b) == 5
      vals = Nx.to_flat_list(b)
      assert vals == [1, 8, 64, 512, 3072]
    end
  end

  describe "levels/0" do
    test "returns [8,8,8,6,5]" do
      l = FSQ.levels()
      assert Nx.to_flat_list(l) == [8, 8, 8, 6, 5]
    end
  end

  describe "bound/2" do
    test "keeps values in level range" do
      z = Nx.tensor([[[1.0, 2.0, 0.0, -1.0, 0.5]]])
      out = FSQ.bound(z)
      assert Nx.shape(out) == {1, 1, 5}
      assert Nx.all(Nx.less_equal(Nx.abs(out), 10)) |> Nx.to_number() == 1
    end

    test "accepts custom eps" do
      z = Nx.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], type: {:f, 32})
      out = FSQ.bound(z, 0.01)
      assert Nx.shape(out) == {1, 1, 5}
      assert Nx.all(Nx.less_equal(Nx.abs(out), 10)) |> Nx.to_number() == 1
    end
  end

  describe "quantize/1" do
    test "returns normalized codes in [-1, 1] range" do
      z = Nx.iota({2, 4, 5}) |> Nx.divide(20) |> Nx.subtract(0.5)
      out = FSQ.quantize(z)
      assert Nx.shape(out) == {2, 4, 5}
      assert Nx.all(Nx.greater_equal(out, -1.1)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(out, 1.1)) |> Nx.to_number() == 1
    end
  end

  describe "codes_to_indices/1" do
    test "returns integer indices in 0..15359" do
      codes = Nx.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], type: {:f, 32})
      idx = FSQ.codes_to_indices(codes)
      assert Nx.shape(idx) == {1, 1}
      val = Nx.to_flat_list(idx) |> List.first()
      assert val >= 0 and val < 15_360
    end

    test "clips indices to vocab 0..15359 (max index 15359)" do
      # codes at upper bound (1.0) exercise scale_and_shift and clip to max index 15359
      codes_high = Nx.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0]]], type: {:f, 32})
      idx = FSQ.codes_to_indices(codes_high)
      vals = Nx.to_flat_list(idx)
      assert Enum.all?(vals, fn v -> v >= 0 and v <= 15_359 end)
    end
  end

  describe "indices_to_codes round-trip" do
    test "indices -> 5-dim codes -> codes_to_indices recovers indices" do
      indices = Nx.tensor([[0, 1, 100, 1000]], type: {:s, 32})
      b = Nx.reshape(FSQ.basis(), {1, 1, 5})
      l = Nx.reshape(FSQ.levels(), {1, 1, 5})
      codes_non_centered = Nx.remainder(Nx.quotient(Nx.reshape(indices, {1, 4, 1}), b), l)
      codes = FSQ.scale_and_shift_inverse(codes_non_centered)
      recovered = FSQ.codes_to_indices(codes)
      assert Nx.shape(recovered) == {1, 4}
      assert Nx.all(Nx.equal(indices, recovered)) |> Nx.to_number() == 1
    end
  end

  describe "encode/2" do
    test "returns {quant_embeds, quant_indices} with correct shapes" do
      project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.subtract(0.05)
      project_in_b = Nx.broadcast(0.0, {5})
      project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.subtract(0.05)
      project_out_b = Nx.broadcast(0.0, {192})

      params = %{
        "project_in" => %{"kernel" => project_in_k, "bias" => project_in_b},
        "project_out" => %{"kernel" => project_out_k, "bias" => project_out_b}
      }

      z = Nx.iota({3, 4, 192}) |> Nx.divide(1000) |> Nx.subtract(0.2)
      {quant_embeds, quant_indices} = FSQ.encode(z, params)
      assert Nx.shape(quant_embeds) == {3, 4, 192}
      assert Nx.shape(quant_indices) == {3, 4}
      assert Nx.all(Nx.greater_equal(quant_indices, 0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less(quant_indices, 15_360)) |> Nx.to_number() == 1
    end
  end

  describe "constants" do
    test "vocab_size, padding_id, seq_len, dim" do
      assert FSQ.vocab_size() == 15_360
      assert FSQ.padding_id() == 15_360
      assert FSQ.seq_len() == 4
      assert FSQ.dim() == 192
    end
  end

  describe "round_ste/1" do
    test "rounds with straight-through gradient shape" do
      z = Nx.tensor([[1.4, 2.6]])
      out = FSQ.round_ste(z)
      assert Nx.shape(out) == {1, 2}
      # Values should be close to rounded
      assert Nx.all(Nx.less_equal(Nx.abs(Nx.subtract(out, Nx.round(z))), 1.0e-5)) |> Nx.to_number() == 1
    end
  end

  describe "scale_and_shift/1" do
    test "maps normalized to level indices" do
      z = Nx.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], type: {:f, 32})
      out = FSQ.scale_and_shift(z)
      assert Nx.shape(out) == {1, 1, 5}
      # Middle of each level
      vals = Nx.to_flat_list(out)
      assert length(vals) == 5
    end
  end

  describe "scale_and_shift and scale_and_shift_inverse round-trip" do
    test "scale_and_shift_inverse(scale_and_shift(z)) recovers z" do
      z = Nx.tensor([[[-0.5, 0.0, 0.5, 1.0, -1.0]]], type: {:f, 32})
      shifted = FSQ.scale_and_shift(z)
      recovered = FSQ.scale_and_shift_inverse(shifted)
      assert Nx.all_close(z, recovered) |> Nx.to_number() == 1
    end
  end

  describe "encode/2 with nil bias" do
    test "works when project_in and project_out have no bias" do
      project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5)
      project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192)
      params = %{
        "project_in" => %{"kernel" => project_in_k, "bias" => nil},
        "project_out" => %{"kernel" => project_out_k, "bias" => nil}
      }
      z = Nx.iota({1, 4, 192}) |> Nx.divide(1000)
      {embeds, indices} = FSQ.encode(z, params)
      assert Nx.shape(embeds) == {1, 4, 192}
      assert Nx.shape(indices) == {1, 4}
    end
  end

  describe "load_params/1" do
    test "accepts project_in/kernel and project_out/kernel keys" do
      k_in = Nx.iota({192, 5}) |> Nx.divide(1)
      k_out = Nx.iota({5, 192}) |> Nx.divide(1)
      params = FSQ.load_params(%{
        "project_in/kernel" => k_in,
        "project_in/bias" => nil,
        "project_out/kernel" => k_out,
        "project_out/bias" => nil
      })
      assert map_size(params["project_in"]) == 2
      assert map_size(params["project_out"]) == 2
    end

    test "transposes project_in when shape is {5, 192}" do
      k_in = Nx.iota({5, 192}) |> Nx.divide(1)
      params = FSQ.load_params(%{
        "fsq.project_in.weight" => k_in,
        "fsq.project_in.bias" => nil,
        "fsq.project_out.weight" => Nx.iota({192, 5}),
        "fsq.project_out.bias" => nil
      })
      assert Nx.shape(params["project_in"]["kernel"]) == {192, 5}
    end

    test "keeps project_out kernel shape {5, 192} as-is" do
      k_out = Nx.iota({5, 192}) |> Nx.divide(1)
      params = FSQ.load_params(%{
        "project_in/kernel" => Nx.iota({192, 5}),
        "project_in/bias" => nil,
        "project_out/kernel" => k_out,
        "project_out/bias" => nil
      })
      assert Nx.shape(params["project_out"]["kernel"]) == {5, 192}
    end

    test "indices_to_codes with loaded params" do
      project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5)
      project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192)
      params = %{
        "project_in" => %{"kernel" => project_in_k, "bias" => nil},
        "project_out" => %{"kernel" => project_out_k, "bias" => nil}
      }
      indices = Nx.tensor([[10, 20, 100, 1000]], type: {:s, 32})
      codes = FSQ.indices_to_codes(indices, params)
      assert Nx.shape(codes) == {1, 4, 192}
    end
  end
end
