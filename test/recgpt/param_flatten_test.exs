# RecGPT.ParamFlatten: flatten, unflatten, spec_from_params, update_params.
defmodule RecGPT.ParamFlattenTest do
  use ExUnit.Case, async: true

  alias RecGPT.ParamFlatten

  defp canonical_keys do
    ["wte", "ae.linear.weight", "ae.linear.bias", "pred_head.weight", "pred_head.bias"]
  end

  defp stub_params do
    %{
      "wte" => Nx.iota({100, 8}) |> Nx.as_type({:f, 32}),
      "ae.linear.weight" => Nx.iota({768, 192}) |> Nx.divide(768 * 192) |> Nx.as_type({:f, 32}),
      "ae.linear.bias" => Nx.broadcast(0.1, {768}) |> Nx.as_type({:f, 32}),
      "pred_head.weight" =>
        Nx.iota({768, 15_361}) |> Nx.divide(768 * 15_361) |> Nx.as_type({:f, 32}),
      "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    }
  end

  describe "spec_from_params/3" do
    test "returns {:ok, spec, total} for params with all canonical keys" do
      params = stub_params()
      assert {:ok, spec, total} = ParamFlatten.spec_from_params(params, canonical_keys())

      assert length(spec) == 5
      assert total == 100 * 8 + 768 * 192 + 768 + 768 * 15_361 + 15_361

      keys = Enum.map(spec, & &1.key) |> Enum.sort()
      assert keys == canonical_keys() |> Enum.sort()

      wte_spec = Enum.find(spec, fn s -> s.key == "wte" end)
      assert wte_spec.shape == {100, 8}
      assert wte_spec.offset == 0
      ae_spec = Enum.find(spec, fn s -> s.key == "ae.linear.weight" end)
      assert ae_spec.shape == {768, 192}
    end

    test "resolves alias: wte from gpt2model.wte" do
      params = %{"gpt2model.wte" => Nx.iota({50, 4}) |> Nx.as_type({:f, 32})}
      keys = ["wte"]
      assert {:ok, [spec], total} = ParamFlatten.spec_from_params(params, keys)
      assert spec.key == "wte"
      assert spec.shape == {50, 4}
      assert total == 200
    end

    test "returns error when a required key is missing" do
      params = Map.drop(stub_params(), ["pred_head.bias"])
      assert {:error, msg} = ParamFlatten.spec_from_params(params, canonical_keys())
      assert msg =~ "missing param"
      assert msg =~ "pred_head.bias"
    end
  end

  describe "flatten/3" do
    test "returns {:ok, flat_tensor, spec} with correct total size" do
      params = stub_params()
      assert {:ok, flat, spec} = ParamFlatten.flatten(params, canonical_keys())

      expected_size = Enum.reduce(spec, 0, fn s, acc -> acc + s.size end)
      assert Nx.shape(flat) == {expected_size}
      assert Nx.size(flat) == expected_size
    end

    test "returns error when params are incomplete" do
      params = %{"wte" => Nx.iota({2, 2}) |> Nx.as_type({:f, 32})}
      assert {:error, _} = ParamFlatten.flatten(params, canonical_keys())
    end
  end

  describe "unflatten/2" do
    test "round-trip: flatten then unflatten recovers same shapes and keys" do
      params = stub_params()
      assert {:ok, flat, spec} = ParamFlatten.flatten(params, canonical_keys())
      unflattened = ParamFlatten.unflatten(flat, spec)

      assert Map.keys(unflattened) |> Enum.sort() == canonical_keys() |> Enum.sort()

      for key <- canonical_keys() do
        orig = params[key]
        restored = unflattened[key]

        assert Nx.shape(restored) == Nx.shape(orig),
               "key #{key}: shape mismatch #{inspect(Nx.shape(restored))} != #{inspect(Nx.shape(orig))}"
      end
    end

    test "unflatten with modified flat tensor changes values" do
      params = stub_params()
      {:ok, flat, spec} = ParamFlatten.flatten(params, canonical_keys())
      # Add 1 to first 10 elements
      add_one = Nx.tensor(List.duplicate(1.0, 10) ++ List.duplicate(0.0, Nx.size(flat) - 10))
      modified = Nx.add(flat, add_one)
      unflattened = ParamFlatten.unflatten(modified, spec)

      # First key is wte with 100*8 elements; first 10 should differ
      wte_orig = params["wte"]
      wte_new = unflattened["wte"]
      refute Nx.all_close(wte_orig, wte_new) |> Nx.to_number() == 1
    end
  end

  describe "update_params/3" do
    test "merges unflattened map into params using checkpoint key names" do
      params = stub_params()
      {:ok, flat, spec} = ParamFlatten.flatten(params, canonical_keys())
      bumped = Nx.add(flat, 0.5)
      unflattened = ParamFlatten.unflatten(bumped, spec)

      # Use key_to_checkpoint so we update the same keys as in params ("wte", not "gpt2model.wte")
      key_to_checkpoint = %{
        "wte" => ["wte"],
        "ae.linear.weight" => ["ae.linear.weight"],
        "ae.linear.bias" => ["ae.linear.bias"],
        "pred_head.weight" => ["pred_head.weight"],
        "pred_head.bias" => ["pred_head.bias"]
      }

      updated = ParamFlatten.update_params(params, unflattened, key_to_checkpoint)

      assert Map.has_key?(updated, "wte")
      assert Map.has_key?(updated, "pred_head.weight")
      assert Nx.all_close(updated["wte"], unflattened["wte"]) |> Nx.to_number() == 1
    end

    test "key_to_checkpoint overrides output key for updated params" do
      params = %{"wte" => Nx.iota({2, 2}) |> Nx.as_type({:f, 32})}
      keys = ["wte"]
      assert {:ok, flat, spec} = ParamFlatten.flatten(params, keys)
      unflattened = ParamFlatten.unflatten(flat, spec)
      # Map canonical "wte" to a different checkpoint key in the output
      key_to_checkpoint = %{"wte" => ["other_key"]}
      updated = ParamFlatten.update_params(params, unflattened, key_to_checkpoint)
      assert Map.has_key?(updated, "other_key")
      assert Nx.all_close(updated["other_key"], unflattened["wte"]) |> Nx.to_number() == 1
    end
  end
end
