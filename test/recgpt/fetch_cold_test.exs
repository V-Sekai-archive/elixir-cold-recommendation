# Cold splits: compute_cold_splits and that cold_test_cases have next_item in cold set.
defmodule RecGPT.FetchColdTest do
  use ExUnit.Case, async: true

  alias RecGPT.Clickstream.Fetch

  describe "compute_cold_splits/4" do
    test "cold set is non-empty for reasonable threshold and cold_test_cases only have next_item in cold set" do
      # Items 0..4. Train: item 0,1,2 appear in 3 sessions each; items 3,4 in 2 each. Cold (<=2): 3, 4.
      train_sequences = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 4],
        [2],
        [3],
        [4]
      ]

      test_cases = [
        %{"context" => [0], "next_item" => 3},
        %{"context" => [1], "next_item" => 4},
        %{"context" => [2], "next_item" => 0}
      ]

      num_items = 5
      max_sessions = 2

      # Session counts in train: 0->3, 1->3, 2->3, 3->2, 4->2. Cold (<=2): 3, 4
      {cold_set, cold_test_cases, cold_train_sequences} =
        Fetch.compute_cold_splits(train_sequences, test_cases, num_items, max_sessions)

      assert MapSet.size(cold_set) >= 1
      assert MapSet.member?(cold_set, 3)
      assert MapSet.member?(cold_set, 4)

      for tc <- cold_test_cases do
        assert MapSet.member?(cold_set, tc["next_item"]),
               "cold_test_cases must only include test cases where next_item is in cold set, got next_item=#{tc["next_item"]}"
      end

      # cold_train_sequences: sequences that contain at least one cold item
      assert is_list(cold_train_sequences)

      for seq <- cold_train_sequences do
        assert Enum.any?(seq, &MapSet.member?(cold_set, &1)),
               "each cold_train sequence must contain at least one cold item"
      end
    end

    test "cold_test_cases shape matches test_cases (context, next_item)" do
      train_sequences = [[0, 1], [1, 2], [2, 0]]

      test_cases = [
        %{"context" => [0], "next_item" => 1},
        %{"context" => [1], "next_item" => 2}
      ]

      num_items = 3

      {_cold_set, cold_test_cases, cold_train_sequences} =
        Fetch.compute_cold_splits(train_sequences, test_cases, num_items, 1)

      for tc <- cold_test_cases do
        assert Map.has_key?(tc, "context")
        assert Map.has_key?(tc, "next_item")
      end

      assert is_list(cold_train_sequences)
      assert Enum.all?(cold_train_sequences, &is_list/1)
    end
  end
end
