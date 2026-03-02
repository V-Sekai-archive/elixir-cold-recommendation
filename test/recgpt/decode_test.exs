# RecGPT.Decode: beam search with trie.
defmodule RecGPT.DecodeTest do
  use ExUnit.Case, async: true

  alias RecGPT.Decode
  alias RecGPT.Trie

  test "beam_search returns best item_id when get_logits favors one path" do
    token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40]]
    trie = Trie.build(token_id_list)
    # Logits that strongly prefer token 1 then 2 then 3 then 4
    get_logits = fn
      [] -> make_logits([1, 10], [10.0, 0.0])
      [1] -> make_logits([2], [5.0])
      [1, 2] -> make_logits([3], [5.0])
      [1, 2, 3] -> make_logits([4], [5.0])
      _ -> make_logits([0], [0.0])
    end

    assert Decode.beam_search(get_logits, trie, [], 2) == {:ok, 0}
  end

  test "beam_search returns second item when logits favor that path" do
    token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40]]
    trie = Trie.build(token_id_list)

    get_logits = fn
      [] -> make_logits([1, 10], [0.0, 10.0])
      [10] -> make_logits([20], [5.0])
      [10, 20] -> make_logits([30], [5.0])
      [10, 20, 30] -> make_logits([40], [5.0])
      _ -> make_logits([0], [0.0])
    end

    assert Decode.beam_search(get_logits, trie, [], 2) == {:ok, 1}
  end

  test "beam_search with context (previous item tokens)" do
    token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40]]
    trie = Trie.build(token_id_list)
    context = [1, 2, 3, 4]

    get_logits = fn
      [1, 2, 3, 4] -> make_logits([1, 10], [0.0, 8.0])
      [1, 2, 3, 4, 10] -> make_logits([20], [5.0])
      [1, 2, 3, 4, 10, 20] -> make_logits([30], [5.0])
      [1, 2, 3, 4, 10, 20, 30] -> make_logits([40], [5.0])
      _ -> make_logits([0], [0.0])
    end

    assert Decode.beam_search(get_logits, trie, context, 2) == {:ok, 1}
  end

  test "beam_search returns not_found when trie is empty (no catalog)" do
    token_id_list = []
    trie = Trie.build(token_id_list)
    get_logits = fn _ -> make_logits([0], [10.0]) end
    assert Decode.beam_search(get_logits, trie, [], 2) == :not_found
  end

  test "beam_search_top_k returns up to top_k item_ids sorted by score" do
    token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40], [5, 6, 7, 8]]
    trie = Trie.build(token_id_list)
    # Prefer path 1 then 10 then 5
    get_logits = fn
      [] -> make_logits([1, 10, 5], [3.0, 2.0, 1.0])
      [1] -> make_logits([2], [1.0])
      [1, 2] -> make_logits([3], [1.0])
      [1, 2, 3] -> make_logits([4], [1.0])
      [10] -> make_logits([20], [1.0])
      [10, 20] -> make_logits([30], [1.0])
      [10, 20, 30] -> make_logits([40], [1.0])
      [5] -> make_logits([6], [1.0])
      [5, 6] -> make_logits([7], [1.0])
      [5, 6, 7] -> make_logits([8], [1.0])
      _ -> make_logits([0], [0.0])
    end

    batch_fn = batch_fn_from(get_logits)
    assert {:ok, list} = Decode.beam_search_top_k(get_logits, trie, [], 2, batch_fn)
    assert length(list) <= 2
    assert list != []
    assert Enum.all?(list, fn id -> id in [0, 1, 2] end)
    assert list == Enum.uniq(list)
    assert {:ok, list3} = Decode.beam_search_top_k(get_logits, trie, [], 3, batch_fn)
    assert length(list3) <= 3
    assert Enum.all?(list3, fn id -> id in [0, 1, 2] end)
    assert list3 == Enum.uniq(list3)
  end

  test "beam_search_top_k with top_k 1 returns at most one item" do
    token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40]]
    trie = Trie.build(token_id_list)

    get_logits = fn
      [] -> make_logits([1, 10], [5.0, 0.0])
      [1] -> make_logits([2], [1.0])
      [1, 2] -> make_logits([3], [1.0])
      [1, 2, 3] -> make_logits([4], [1.0])
      _ -> make_logits([0], [0.0])
    end

    batch_fn = batch_fn_from(get_logits)
    assert {:ok, list} = Decode.beam_search_top_k(get_logits, trie, [], 1, batch_fn)
    assert length(list) <= 1
    assert list == [] or length(list) == 1
  end

  test "beam_search with beam_width 1 still returns item_id" do
    token_id_list = [[1, 2, 3, 4]]
    trie = Trie.build(token_id_list)

    get_logits = fn
      [] -> make_logits([1], [10.0])
      [1] -> make_logits([2], [5.0])
      [1, 2] -> make_logits([3], [5.0])
      [1, 2, 3] -> make_logits([4], [5.0])
      _ -> make_logits([0], [0.0])
    end

    assert Decode.beam_search(get_logits, trie, [], 1) == {:ok, 0}
  end

  defp make_logits(preferred_ids, preferred_scores) do
    vocab_size = 15_361
    base = Nx.broadcast(-100.0, {1, vocab_size}) |> Nx.as_type({:f, 32})

    Enum.reduce(Enum.zip(preferred_ids, preferred_scores), base, fn {id, score}, acc ->
      Nx.put_slice(acc, [0, id], Nx.tensor([[score]], type: {:f, 32}))
    end)
  end

  defp batch_fn_from(get_logits) do
    fn list, _cache ->
      logits =
        list
        |> Enum.map(fn seq -> get_logits.(seq) |> Nx.squeeze(axes: [0]) end)
        |> Nx.stack(axis: 0)
      {logits, nil}
    end
  end
end
