defmodule RecGPT.FSQEncoderTest do
  use ExUnit.Case, async: true

  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder

  describe "encode_embeddings_to_token_id_list/3" do
    test "returns one 4-token list per item" do
      # 3 items, batch_size 2 so we get 2 batches
      num_items = 3
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items) |> Nx.subtract(0.1)
      params = make_params()
      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 2)
      assert length(token_id_list) == 3
      assert Enum.all?(token_id_list, fn tokens -> length(tokens) == 4 end)

      assert Enum.all?(token_id_list, fn tokens ->
               Enum.all?(tokens, fn t -> is_integer(t) and t >= 0 and t < FSQ.vocab_size() end)
             end)
    end

    test "custom batch_size" do
      embeddings = Nx.iota({2, 768}) |> Nx.divide(768 * 2)
      params = make_params()
      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 1)
      assert length(token_id_list) == 2
    end

    test "single item uses default batch_size" do
      embeddings = Nx.iota({1, 768}) |> Nx.divide(768)
      params = make_params()
      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params)
      assert length(token_id_list) == 1
      assert length(hd(token_id_list)) == 4
    end

    test "multiple batches (5 items, batch_size 2) produces 5 token lists" do
      num_items = 5
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items) |> Nx.subtract(0.05)
      params = make_params()
      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 2)
      assert length(token_id_list) == 5
      assert Enum.all?(token_id_list, fn tokens -> length(tokens) == 4 end)
    end

    test "same embeddings and params yield deterministic token_id_list" do
      embeddings = Nx.iota({2, 768}) |> Nx.divide(768 * 2) |> Nx.subtract(0.1)
      params = make_params()
      first = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 1)
      second = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 1)
      assert first == second
    end
  end

  describe "load_embeddings_from_npy/1" do
    test "raises on missing file" do
      path =
        Path.join(
          System.tmp_dir!(),
          "recgpt_nonexistent_#{:erlang.unique_integer([:positive])}.npy"
        )

      assert_raise RuntimeError, ~r/Failed to load embeddings/, fn ->
        FSQEncoder.load_embeddings_from_npy(path)
      end
    end

    test "loads tensor from valid .npy file" do
      path = Path.join(System.tmp_dir!(), "recgpt_emb_#{:erlang.unique_integer([:positive])}.npy")
      tensor = Nx.iota({2, 768}) |> Nx.divide(768) |> Nx.as_type({:f, 32})

      try do
        Npy.save(tensor, path)
        loaded = FSQEncoder.load_embeddings_from_npy(path)
        assert Nx.shape(loaded) == {2, 768}
        assert Nx.all_close(loaded, tensor) |> Nx.to_number() == 1
      after
        File.rm(path)
      end
    end
  end

  defp make_params do
    project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.subtract(0.05)
    project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.subtract(0.05)

    FSQ.load_params(%{
      "project_in/kernel" => project_in_k,
      "project_in/bias" => nil,
      "project_out/kernel" => project_out_k,
      "project_out/bias" => nil
    })
  end
end
