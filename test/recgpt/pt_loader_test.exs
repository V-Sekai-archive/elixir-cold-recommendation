# RecGPT.PtLoader: load zip-format .pt in pure Elixir (Unpickler + Unzip).
defmodule RecGPT.PtLoaderTest do
  use ExUnit.Case, async: true

  alias RecGPT.PtLoader

  @fixture_path Path.join([File.cwd!(), "test", "fixtures", "sample.pt"])

  @tag :pt_fixture
  test "load! loads sample.pt fixture and returns state_dict of Nx tensors" do
    unless File.regular?(@fixture_path) do
      raise "Missing fixture: run python scripts/generate_pt_fixture.py to create test/fixtures/sample.pt"
    end

    params = PtLoader.load!(@fixture_path)

    assert map_size(params) >= 1
    assert Enum.all?(params, fn {_k, v} -> is_struct(v, Nx.Tensor) end)
  end

  test "load! raises ArgumentError for non-zip file" do
    dir = Path.join(System.tmp_dir!(), "recgpt_pt_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    path = Path.join(dir, "not_a_zip.pt")
    File.write!(path, "not a zip file content")

    try do
      assert_raise ArgumentError, ~r/only supports zip-based .pt|Got non-zip/, fn ->
        PtLoader.load!(path)
      end
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
