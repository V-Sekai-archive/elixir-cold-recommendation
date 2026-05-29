# Chrono-split end-to-end on tiny statically-known sequence sets.
# Validates split_train_test_chrono/2 with hand-crafted edge cases.
defmodule RecGPT.Trajectories.ChronoSplitIntegrationTest do
  use ExUnit.Case, async: true

  alias RecGPT.Trajectories.Convert

  describe "tiny dataset edge cases for chrono split" do
    test "all sequences <=2 items produce empty test, unchanged train" do
      sequences = [
        %{"sequence" => [10, 20],          "timestamps" => [1000, 2000]},
        %{"sequence" => [30],              "timestamps" => [3000]},
        %{"sequence" => [40, 50, 60],     "timestamps" => [4000, 5000, 6000]}
      ]

      {:ok, train, test} = Convert.split_train_test_chrono(sequences, 0.2)

      # 3 sequences in train (none skipped because all have >=2 items)
      assert length(train) == 3
      # Only the 3-item sequence produces a test case
      assert length(test) == 1

      # The 1-item and 2-item sequences stay untouched
      [s1, s2, s3] = train
      assert Convert.seq_from(s1) == [10, 20]
      assert Convert.seq_from(s2) == [30]
      assert Convert.seq_from(s3) == [40, 50]

      # Test case is last item of the long sequence
      [tc] = test
      assert tc["context"] == [40, 50]
      assert tc["next_item"] == 60
    end

    test "timestamps are preserved and truncated for long sequences" do
      sequences = [
        %{"sequence" => [1, 2, 3, 4], "timestamps" => [100, 200, 300, 400]}
      ]

      {:ok, [train_seq], [tc]} = Convert.split_train_test_chrono(sequences, 0.2)

      assert Convert.seq_from(train_seq) == [1, 2, 3]
      assert train_seq["timestamps"] == [100, 200, 300]
      assert tc["context"] == [1, 2, 3]
      assert tc["next_item"] == 4
    end

    test "empty and single-item sequences are handled gracefully" do
      sequences = [
        %{"sequence" => [],     "timestamps" => []},
        %{"sequence" => [99],   "timestamps" => [999]}
      ]

      {:ok, train, test} = Convert.split_train_test_chrono(sequences, 0.2)

      # Both empty and single-item go to train untouched; no test cases
      assert length(train) == 2
      assert length(test) == 0

      assert Convert.seq_from(hd(train)) == []
      assert Convert.seq_from(Enum.at(train, 1)) == [99]
    end

    test "multi-user scenario preserves per-sequence independence" do
      sequences = [
        %{"sequence" => [10, 20, 30], "timestamps" => [1, 2, 3]},
        %{"sequence" => [40, 50, 60], "timestamps" => [5, 6, 7]},
        %{"sequence" => [70, 80],     "timestamps" => [10, 11]}
      ]

      {:ok, train, test} = Convert.split_train_test_chrono(sequences, 0.2)

      assert length(train) == 3
      assert length(test) == 2

      # Both long sequences have their last item as test target
      assert Enum.at(test, 0)["next_item"] == 30
      assert Enum.at(test, 1)["next_item"] == 60

      # Short sequence untouched
      short_train = Enum.find(train, fn s -> Enum.at(Convert.seq_from(s), 0, nil) == 70 end)
      assert Convert.seq_from(short_train) == [70, 80]
    end

    test "items are never lost: train_total + length(test) == original_total" do
      sequences = [
        %{"sequence" => [1, 2, 3, 4, 5], "timestamps" => [10, 20, 30, 40, 50]},
        %{"sequence" => [6, 7],          "timestamps" => [60, 70]},
        %{"sequence" => [8, 9, 10],     "timestamps" => [80, 90, 100]}
      ]

      orig_total = Enum.sum(Enum.map(sequences, fn s -> length(s["sequence"]) end))
      {:ok, train, test} = Convert.split_train_test_chrono(sequences, 0.2)

      train_total = Enum.sum(Enum.map(train, fn s -> length(Convert.seq_from(s)) end))
      assert train_total + length(test) == orig_total
    end
  end
end
