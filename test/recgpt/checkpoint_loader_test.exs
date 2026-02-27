# RecGPT.CheckpointLoader: load_from_export with manifest + .npy files.
defmodule RecGPT.CheckpointLoaderTest do
  use ExUnit.Case, async: true

  alias RecGPT.CheckpointLoader

  @tag :integration
  @tag timeout: 30_000
  test "load_from_export loads tensors from manifest and .npy files" do
    dir = Path.join(System.tmp_dir!(), "recgpt_ckpt_#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    manifest = %{
      "wte" => %{"file" => "wte.npy", "shape" => [2, 8]},
      "pred_head.weight" => %{"file" => "pred_head_weight.npy", "shape" => [10, 8]}
    }

    wte = Nx.iota({2, 8}) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({10, 8}) |> Nx.divide(80) |> Nx.as_type({:f, 32})

    try do
      File.write!(Path.join(dir, "manifest.json"), Jason.encode!(manifest))
      Npy.save(wte, Path.join(dir, "wte.npy"))
      Npy.save(head_w, Path.join(dir, "pred_head_weight.npy"))

      params = CheckpointLoader.load_from_export(dir)
      assert map_size(params) == 2
      assert Nx.shape(params["wte"]) == {2, 8}
      assert Nx.shape(params["pred_head.weight"]) == {10, 8}
      assert Nx.all_close(params["wte"], wte) |> Nx.to_number() == 1
    after
      File.rm_rf(dir)
    end
  end

  test "load_from_export raises when manifest.json is missing" do
    dir = Path.join(System.tmp_dir!(), "recgpt_empty_#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      assert_raise File.Error, fn ->
        CheckpointLoader.load_from_export(dir)
      end
    after
      File.rm_rf(dir)
    end
  end

  test "load_from_export raises when manifest references missing .npy file" do
    dir =
      Path.join(System.tmp_dir!(), "recgpt_missing_npy_#{:erlang.unique_integer([:positive])}")

    File.mkdir_p!(dir)
    manifest = %{"wte" => %{"file" => "nonexistent.npy", "shape" => [2, 8]}}
    File.write!(Path.join(dir, "manifest.json"), Jason.encode!(manifest))

    try do
      assert_raise RuntimeError, ~r/Failed to load|nonexistent/, fn ->
        CheckpointLoader.load_from_export(dir)
      end
    after
      File.rm_rf(dir)
    end
  end
end
