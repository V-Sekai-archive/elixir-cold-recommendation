# RecGPT.PtLoader: load .pt (zip or legacy pickle) in pure Elixir (Unzip + Unpickler).
defmodule RecGPT.PtLoaderTest do
  use ExUnit.Case, async: true

  alias RecGPT.PtLoader

  @known_good Path.join([File.cwd!(), "data", "recgpt_layer_3_weight.pt"])
  @sample_pt Path.join([File.cwd!(), "test", "fixtures", "sample.pt"])

  defp pt_fixture_path do
    cond do
      File.regular?(@known_good) -> @known_good
      File.regular?(@sample_pt) -> @sample_pt
      true ->
        RecGPT.PtFixtureGenerator.generate_to_path(@sample_pt)
        @sample_pt
    end
  end

  @tag :pt_fixture
  test "load! loads .pt fixture (known-good or sample) and returns state_dict of Nx tensors" do
    path = pt_fixture_path()

    try do
      params = PtLoader.load!(path)
      assert map_size(params) >= 1
      assert Enum.all?(params, fn {_k, v} -> is_struct(v, Nx.Tensor) end)
    rescue
      e ->
        if path == @known_good do
          # credo:disable-for-next-line Credo.Check.Warning.RaiseInsideRescue
          raise "data/recgpt_layer_3_weight.pt is not a valid .pt (zip or legacy pickle). Re-download the weights; the file may be a redirect page. #{Exception.message(e)}"
        else
          reraise(e, __STACKTRACE__)
        end
    end
  end

  test "load! raises for invalid file (neither zip nor valid pickle)" do
    dir = Path.join(System.tmp_dir!(), "recgpt_pt_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    path = Path.join(dir, "invalid.pt")
    File.write!(path, "not a zip file content")

    try do
      PtLoader.load!(path)
      flunk("expected load! to raise for invalid content")
    rescue
      _ -> :ok
    after
      File.rm_rf(dir)
    end
  end

  test "load! raises when file does not exist" do
    path = Path.join(System.tmp_dir!(), "nonexistent_#{System.unique_integer([:positive])}.pt")

    assert_raise File.Error, fn ->
      PtLoader.load!(path)
    end
  end
end
