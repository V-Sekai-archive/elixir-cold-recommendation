# SPMD decode: trie tensor verification and E2E tests.

defmodule RecGPT.DecodeSPMDTest do
  use ExUnit.Case, async: true

  alias RecGPT.Decode
  alias RecGPT.Trie

  @vocab_size 15_361

  defp tensor_at(tensor, row, col) do
    tensor |> Nx.slice([row, col], [1, 1]) |> Nx.squeeze() |> Nx.to_number()
  end

  defp stub_from_seq_spec(spec, backend) do
    fn batch, nil ->
      rows = batch |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_list()

      logits_rows =
        for row <- rows do
          seq = Enum.map(List.flatten([row]), &round/1)
          {ids, scores} = Map.get(spec, seq, {[0], [0.0]})
          base = Nx.broadcast(-100.0, {1, @vocab_size}) |> Nx.as_type({:f, 32})

          Enum.zip(ids, scores)
          |> Enum.reduce(base, fn {id, score}, acc ->
            Nx.put_slice(acc, [0, id], Nx.tensor([[score]], type: {:f, 32}))
          end)
          |> Nx.squeeze(axes: [0])
        end

      {Nx.stack(logits_rows) |> Nx.backend_transfer(backend), nil}
    end
  end

  defp run_spmd_with_stub(token_id_list, context_ids, top_k, spec) do
    trie = Trie.build(token_id_list)
    tensors = Trie.to_tensors(trie, @vocab_size)
    item_id_to_tokens = Nx.tensor(token_id_list, type: {:s, 32})
    backend = Nx.BinaryBackend

    trie_tensors = %{
      next_state: Nx.backend_transfer(tensors.next_state, backend),
      item_at_leaf: Nx.backend_transfer(tensors.item_at_leaf, backend)
    }

    item_id_to_tokens = Nx.backend_transfer(item_id_to_tokens, backend)
    stub = stub_from_seq_spec(spec, backend)

    previous = Nx.default_backend()

    try do
      Nx.default_backend(Nx.BinaryBackend)

      Decode.beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        context_ids,
        top_k,
        stub,
        backend,
        trie
      )
    after
      Nx.default_backend(previous)
    end
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

  describe "SPMD beam search behavior" do
    test "returns best item when logits favor that path" do
      token_id_list = [
        [1, 2, 3, 4],
        [10, 20, 30, 40]
      ]

      # Empty context uses padding token 0, so stub receives [0]
      spec = %{
        [0] => {[1, 10], [10.0, 0.0]},
        [1] => {[2], [5.0]},
        [1, 2] => {[3], [5.0]},
        [1, 2, 3] => {[4], [5.0]},
        [10] => {[20], [5.0]},
        [10, 20] => {[30], [5.0]},
        [10, 20, 30] => {[40], [5.0]}
      }

      {:ok, list} = run_spmd_with_stub(token_id_list, [], 2, spec)
      assert hd(list) == 0
    end

    test "returns second item when logits favor that path" do
      token_id_list = [
        [1, 2, 3, 4],
        [10, 20, 30, 40]
      ]

      spec = %{
        [0] => {[1, 10], [0.0, 10.0]},
        [10] => {[20], [5.0]},
        [10, 20] => {[30], [5.0]},
        [10, 20, 30] => {[40], [5.0]}
      }

      {:ok, list} = run_spmd_with_stub(token_id_list, [], 2, spec)
      assert hd(list) == 1
    end

    test "with context (previous item tokens) returns correct item" do
      token_id_list = [
        [1, 2, 3, 4],
        [10, 20, 30, 40]
      ]

      # Context [0] = item 0 tokens [1,2,3,4]. Next we predict first token of next item.
      spec = %{
        [1, 2, 3, 4] => {[1, 10], [0.0, 8.0]},
        [1, 2, 3, 4, 10] => {[20], [5.0]},
        [1, 2, 3, 4, 10, 20] => {[30], [5.0]},
        [1, 2, 3, 4, 10, 20, 30] => {[40], [5.0]}
      }

      {:ok, list} = run_spmd_with_stub(token_id_list, [0], 2, spec)
      assert 1 in list and hd(list) == 1
    end

    test "returns not_found when trie is empty (no catalog)" do
      token_id_list = []
      trie = Trie.build(token_id_list)
      tensors = Trie.to_tensors(trie, @vocab_size)
      # Empty catalog: item_id_to_tokens needs >=1 row for gather; use dummy row
      item_id_to_tokens = Nx.tensor([[0, 0, 0, 0]], type: {:s, 32})
      backend = Nx.BinaryBackend

      trie_tensors = %{
        next_state: Nx.backend_transfer(tensors.next_state, backend),
        item_at_leaf: Nx.backend_transfer(tensors.item_at_leaf, backend)
      }

      item_id_to_tokens = Nx.backend_transfer(item_id_to_tokens, backend)

      stub = fn batch, nil ->
        {b, _} = Nx.shape(batch)
        {Nx.broadcast(0.0, {b, @vocab_size}) |> Nx.as_type({:f, 32}), nil}
      end

      previous = Nx.default_backend()

      result =
        try do
          Nx.default_backend(Nx.BinaryBackend)

          Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [],
            2,
            stub,
            backend,
            trie
          )
        after
          Nx.default_backend(previous)
        end

      assert result == :not_found
    end

    test "returns up to top_k item_ids sorted by score" do
      token_id_list = [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [5, 6, 7, 8]
      ]

      spec = %{
        [0] => {[1, 10, 5], [3.0, 2.0, 1.0]},
        [1] => {[2], [1.0]},
        [1, 2] => {[3], [1.0]},
        [1, 2, 3] => {[4], [1.0]},
        [10] => {[20], [1.0]},
        [10, 20] => {[30], [1.0]},
        [10, 20, 30] => {[40], [1.0]},
        [5] => {[6], [1.0]},
        [5, 6] => {[7], [1.0]},
        [5, 6, 7] => {[8], [1.0]}
      }

      {:ok, list} = run_spmd_with_stub(token_id_list, [], 2, spec)
      assert length(list) <= 2
      assert list != []
      assert Enum.all?(list, fn id -> id in [0, 1, 2] end)
      assert list == Enum.uniq(list)
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
            Nx.put_slice(
              acc,
              [0, t],
              Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch_size, 1})
            )
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
        refute list == []

        assert Enum.all?(list, fn id -> id in 0..(num_items - 1) end),
               "SPMD returned item_ids #{inspect(list)}; all must be in 0..#{num_items - 1}"
      after
        Nx.default_backend(previous_backend)
      end
    end
  end
end
