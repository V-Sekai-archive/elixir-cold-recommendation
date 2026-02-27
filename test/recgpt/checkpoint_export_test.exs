# RecGPT.CheckpointExport: write manifest + .npy; round-trip with CheckpointLoader.
defmodule RecGPT.CheckpointExportTest do
  use ExUnit.Case, async: true

  alias RecGPT.CheckpointExport
  alias RecGPT.CheckpointLoader

  defp dummy_params do
    %{
      "wte" => Nx.iota({100, 8}) |> Nx.as_type({:f, 32}),
      "pred_head.weight" => Nx.iota({8, 100}) |> Nx.as_type({:f, 32}),
      "pred_head.bias" => Nx.broadcast(0.0, {100}) |> Nx.as_type({:f, 32})
    }
  end

  test "write_export creates manifest.json and .npy files" do
    params = dummy_params()
    dir = Path.join(System.tmp_dir!(), "recgpt_export_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      :ok = CheckpointExport.write_export(params, dir)

      assert File.regular?(Path.join(dir, "manifest.json"))
      manifest = File.read!(Path.join(dir, "manifest.json")) |> Jason.decode!()
      assert map_size(manifest) == 3
      assert manifest["wte"]["file"] == "wte.npy"
      assert manifest["wte"]["shape"] == [100, 8]
      assert File.regular?(Path.join(dir, "wte.npy"))
      assert File.regular?(Path.join(dir, "pred_head.weight.npy"))
      assert File.regular?(Path.join(dir, "pred_head.bias.npy"))
    after
      File.rm_rf(dir)
    end
  end

  test "write_export raises when a value is not an Nx.Tensor" do
    params = %{
      "wte" => Nx.iota({2, 2}) |> Nx.as_type({:f, 32}),
      "bad" => "not a tensor"
    }

    dir = Path.join(System.tmp_dir!(), "recgpt_export_bad_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      assert_raise ArgumentError, ~r/not an Nx.Tensor/, fn ->
        CheckpointExport.write_export(params, dir)
      end
    after
      File.rm_rf(dir)
    end
  end

  test "round-trip: write_export then load_from_export returns same keys and shapes" do
    params = dummy_params()

    dir =
      Path.join(
        System.tmp_dir!(),
        "recgpt_export_roundtrip_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(dir)

    try do
      :ok = CheckpointExport.write_export(params, dir)
      loaded = CheckpointLoader.load_from_export(dir)

      assert Map.keys(loaded) |> Enum.sort() == Map.keys(params) |> Enum.sort()

      for key <- Map.keys(params) do
        assert Nx.shape(loaded[key]) == Nx.shape(params[key])
        assert Nx.all_close(loaded[key], params[key])
      end
    after
      File.rm_rf(dir)
    end
  end
end
