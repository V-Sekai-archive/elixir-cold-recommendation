# RecGPT.Trie: build, lookup, valid_next_tokens.
defmodule RecGPT.TrieTest do
  use ExUnit.Case, async: true

  alias RecGPT.Trie

  test "build from token_id_list and lookup" do
    token_id_list = [
      [1, 2, 3, 4],
      [10, 20, 30, 40],
      [1, 2, 99, 100]
    ]

    trie = Trie.build(token_id_list)
    assert Trie.lookup(trie, [1, 2, 3, 4]) == {:ok, 0}
    assert Trie.lookup(trie, [10, 20, 30, 40]) == {:ok, 1}
    assert Trie.lookup(trie, [1, 2, 99, 100]) == {:ok, 2}
    assert Trie.lookup(trie, [0, 0, 0, 0]) == :not_found
    assert Trie.lookup(trie, [1, 2, 3, 5]) == :not_found
  end

  test "valid_next_tokens at each prefix level" do
    token_id_list = [[1, 2, 3, 4], [1, 2, 99, 100], [5, 6, 7, 8]]
    trie = Trie.build(token_id_list)
    # First token: 1 or 5
    assert Enum.sort(Trie.valid_next_tokens(trie, [])) == [1, 5]
    # After 1: second token 2
    assert Trie.valid_next_tokens(trie, [1]) == [2]
    # After 1,2: third token 3 or 99
    assert Enum.sort(Trie.valid_next_tokens(trie, [1, 2])) == [3, 99]
    # After 1,2,3: fourth token 4
    assert Trie.valid_next_tokens(trie, [1, 2, 3]) == [4]
    # After 1,2,99: fourth token 100
    assert Trie.valid_next_tokens(trie, [1, 2, 99]) == [100]
    # After full sequence: no next
    assert Trie.valid_next_tokens(trie, [1, 2, 3, 4]) == []
    # Invalid prefix
    assert Trie.valid_next_tokens(trie, [0, 0, 0]) == []
  end

  test "build with empty list" do
    trie = Trie.build([])
    assert Trie.lookup(trie, [1, 2, 3, 4]) == :not_found
    assert Trie.valid_next_tokens(trie, []) == []
  end

  test "build skips malformed entries" do
    token_id_list = [[1, 2, 3, 4], [1, 2], [10, 20, 30, 40]]
    trie = Trie.build(token_id_list)
    assert Trie.lookup(trie, [1, 2, 3, 4]) == {:ok, 0}
    assert Trie.lookup(trie, [10, 20, 30, 40]) == {:ok, 2}
    assert Trie.valid_next_tokens(trie, []) == [1, 10]
  end

  test "seq_len is 4" do
    assert Trie.seq_len() == 4
  end

  test "lookup returns :not_found for list length != 4" do
    token_id_list = [[1, 2, 3, 4]]
    trie = Trie.build(token_id_list)
    assert Trie.lookup(trie, [1, 2]) == :not_found
    assert Trie.lookup(trie, [1, 2, 3]) == :not_found
    assert Trie.lookup(trie, [1, 2, 3, 4, 5]) == :not_found
    assert Trie.lookup(trie, []) == :not_found
  end
end
