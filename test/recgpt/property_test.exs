# Property-based tests: trie build/lookup round-trip (see docs/14_top_tier_recommendations.md).
defmodule RecGPT.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties
  alias RecGPT.Trie

  @vocab_max 15_359

  property "trie build and lookup round-trip: item_id maps back via token list" do
    four_tokens =
      StreamData.list_of(StreamData.integer(0..@vocab_max), min_length: 4, max_length: 4)

    token_id_list_gen = StreamData.list_of(four_tokens, min_length: 1, max_length: 50)

    check all(token_id_list <- token_id_list_gen) do
      trie = Trie.build(token_id_list)

      for {tokens, idx} <- Enum.with_index(token_id_list) do
        assert Trie.lookup(trie, tokens) == {:ok, idx}
      end
    end
  end
end
