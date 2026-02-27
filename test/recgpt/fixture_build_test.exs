# FixtureBuild: build_from_embeddings produces valid fixture (num_items, token_id_list shape).
defmodule RecGPT.FixtureBuildTest do
  use ExUnit.Case, async: true

  alias RecGPT.CheckpointExport
  alias RecGPT.FixtureBuild
  alias RecGPT.FSQ

  describe "build_from_embeddings/3" do
    test "produces fixture with num_items and token_id_list of correct length and shape" do
      num_items = 3
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items) |> Nx.as_type({:f, 32})
      ckpt_dir = write_fsq_export!()

      try do
        fixture = FixtureBuild.build_from_embeddings(embeddings, ckpt_dir, [])

        assert fixture["num_items"] == num_items
        assert is_list(fixture["token_id_list"])
        assert length(fixture["token_id_list"]) == num_items

        assert Enum.all?(fixture["token_id_list"], fn tokens ->
                 is_list(tokens) and length(tokens) == 4 and
                   Enum.all?(tokens, fn t -> is_integer(t) and t >= 0 and t < FSQ.vocab_size() end)
               end)
      after
        File.rm_rf(ckpt_dir)
      end
    end
  end

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

  defp write_fsq_export! do
    dir = Path.join(System.tmp_dir!(), "recgpt_fsq_ckpt_#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    # Use keys without "/" so CheckpointExport writes flat filenames (no subdirs)
    project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.as_type({:f, 32})
    project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.as_type({:f, 32})

    params = %{
      "fsq.project_in.weight" => project_in_k,
      "fsq.project_out.weight" => project_out_k
    }

    CheckpointExport.write_export(params, dir)
    dir
  end
end
