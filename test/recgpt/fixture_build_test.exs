# FixtureBuild: write_fixture and build (items → fixture).
defmodule RecGPT.FixtureBuildTest do
  use ExUnit.Case, async: true

  alias RecGPT.FixtureBuild

  describe "write_fixture/2" do
    test "writes JSON that load_fixture can read (num_items, token_id_list)" do
      fixture = %{"num_items" => 2, "token_id_list" => [[1, 2, 3, 4], [5, 6, 7, 8]]}

      path =
        Path.join(System.tmp_dir!(), "recgpt_fixture_#{:erlang.unique_integer([:positive])}.json")

      try do
        :ok = FixtureBuild.write_fixture(fixture, path)
        assert File.regular?(path)
        raw = File.read!(path) |> Jason.decode!()
        assert raw["num_items"] == 2
        assert length(raw["token_id_list"]) == 2
        assert hd(raw["token_id_list"]) == [1, 2, 3, 4]

        # Same shape expected by Serve.load_fixture
        decoded = File.read!(path) |> Jason.decode!()
        assert decoded["num_items"] == 2
        assert length(decoded["token_id_list"]) == 2
      after
        if File.regular?(path), do: File.rm(path)
      end
    end
  end
end
