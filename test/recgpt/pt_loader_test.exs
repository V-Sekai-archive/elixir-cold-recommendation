# RecGPT.PtLoader: load zip-based .pt in pure Elixir (Unzip + Unpickler).
defmodule RecGPT.PtLoaderTest do
  use ExUnit.Case, async: true

  alias RecGPT.PtLoader

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
