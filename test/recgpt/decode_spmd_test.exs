# SPMD decode: trie tensor verification and E2E tests.

defmodule RecGPT.DecodeSPMDTest do
  use ExUnit.Case, async: true

  alias RecGPT.Trie
  alias RecGPT.Decode

  @vocab_size 15_361

  defp tensor_at(tensor, row, col) do
    tensor |> Nx.slice([row, col], [1, 1]) |> Nx.squeeze() |> Nx.to_number()
  end

  defp assert_tensors_match_trie(trie, token_id_list, next_state, item_at_leaf) do
    for {path, expected_item_id} <- Enum.with_index(token_id_list) do
      [t0, t1, t2, t3] = path
      assert {:ok, ^expected_item_id} = Trie.lookup(trie, path)

      s1 = tensor_at(next_state, 0, t0)
      assert s1 >= 0, "next_state[0, #{t0}] = #{s1} for path #{inspect(path)}"

      s2 = tensor_at(next_state, s1, t1)
      assert s2 >= 0, "next_state[#{s1}, #{t1}] = #{s2}"

      s3 = tensor_at(next_state, s2, t2)
      assert s3 >= 0, "next_state[#{s2}, #{t2}] = #{s3}"

      tensor_item = tensor_at(item_at_leaf, s3, t3)
      assert tensor_item == expected_item_id,
             "item_at_leaf[#{s3}, #{t3}] = #{tensor_item}, expected #{expected_item_id}"
    end
  end

  describe "trie tensors match trie map" do
    test "walk next_state and item_at_leaf matches Trie.lookup for all paths" do
      token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40], [1, 2, 99, 100]]
      trie = Trie.build(token_id_list)
      tensors = Trie.to_tensors(trie, @vocab_size)
      assert_tensors_match_trie(trie, token_id_list, tensors.next_state, tensors.item_at_leaf)
    end

    test "trie tensors with fixture (first 20 items)" do
      fixture_path = Path.expand("data/steam/fixture.json", File.cwd!())
      unless File.regular?(fixture_path), do: raise("Fixture not found: #{fixture_path}")

      token_id_list =
        fixture_path
        |> File.read!()
        |> Jason.decode!()
        |> Map.get("token_id_list", [])
        |> Enum.map(&Enum.map(&1, fn x -> round(x) end))
        |> Enum.take(20)

      trie = Trie.build(token_id_list)
      tensors = Trie.to_tensors(trie, @vocab_size)
      assert_tensors_match_trie(trie, token_id_list, tensors.next_state, tensors.item_at_leaf)
    end
  end

  describe "SPMD E2E" do
    test "beam_search_top_k_spmd returns valid item_ids from catalog" do
      token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40], [1, 2, 99, 100]]
      valid_tokens = token_id_list |> List.flatten() |> Enum.uniq()
      num_items = length(token_id_list)
      trie = Trie.build(token_id_list)
      tensors = Trie.to_tensors(trie, @vocab_size)
      item_id_to_tokens = Nx.tensor(token_id_list, type: {:s, 32})

      stub_fn = fn batch, nil ->
        {batch_size, _} = Nx.shape(batch)
        logits = Nx.broadcast(0.0, {batch_size, @vocab_size}) |> Nx.as_type({:f, 32})
        logits =
          Enum.reduce(valid_tokens, logits, fn t, acc ->
            Nx.put_slice(acc, [0, t], Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch_size, 1}))
          end)
        {logits, nil}
      end

      previous_backend = Nx.default_backend()
      try do
        Nx.default_backend(Nx.BinaryBackend)

        backend = Nx.BinaryBackend
        trie_tensors = %{
          next_state: Nx.backend_transfer(tensors.next_state, backend),
          item_at_leaf: Nx.backend_transfer(tensors.item_at_leaf, backend)
        }
        item_id_to_tokens = Nx.backend_transfer(item_id_to_tokens, backend)

        result =
          Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [0],
            3,
            stub_fn,
            backend,
            trie
          )

        assert {:ok, list} = result
        assert length(list) >= 1
        assert Enum.all?(list, fn id -> id in 0..(num_items - 1) end),
               "SPMD returned item_ids #{inspect(list)}; all must be in 0..#{num_items - 1}"
      after
        Nx.default_backend(previous_backend)
      end
    end
  end
end
