defmodule RecGPT.ChronoSplitPropertyTest do
  @moduledoc """
  Property-based tests for temporal split logic using ExUnit + StreamData.
  """
  use ExUnit.Case, async: true
  alias RecGPT.Trajectories.Convert

  @num_runs 200

  defp sequence_gen do
    StreamData.map(
      StreamData.list_of(StreamData.integer(1..100), min_length: 2, max_length: 50),
      fn ids ->
        ts = Enum.map(Enum.with_index(ids), fn {_, i} -> i * 1000 end)
        %{"sequence" => ids, "timestamps" => ts}
      end
    )
  end

  defp dataset_gen do
    StreamData.list_of(sequence_gen(), min_length: 1, max_length: 20)
  end

  test "LIB-4: all originals preserved in train; only long yields test case" do
    for sequences <- Enum.take(dataset_gen(), @num_runs) do
      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      assert length(train_seqs) == length(sequences)

      # Match each train seq with its original by position
      Enum.zip(sequences, train_seqs)
      |> Enum.each(fn {orig, tr} ->
        orig_seq = orig["sequence"]
        tr_seq = Convert.seq_from(tr)

        cond do
          # Short sequences stay intact
          length(orig_seq) <= 2 ->
            assert tr_seq == orig_seq, "short should stay intact"

          # Long sequences get truncated by 1
          true ->
            assert tr_seq == Enum.take(orig_seq, length(orig_seq) - 1),
                   "train should be all-but-last"
        end
      end)

      long_count = Enum.count(sequences, fn s -> length(s["sequence"]) > 2 end)
      assert length(test_cases) == long_count

      Enum.all?(test_cases, fn tc -> length(tc["context"]) > 0 end)
    end
  end

  test "LIB-1: next_item is last item for long sequences" do
    for sequences <- Enum.take(dataset_gen(), @num_runs) do
      {:ok, _, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      long_seqs = Enum.filter(sequences, fn s -> length(s["sequence"]) > 2 end)
      assert length(test_cases) == length(long_seqs)

      Enum.zip(long_seqs, test_cases)
      |> Enum.each(fn {orig, tc} ->
        assert List.last(orig["sequence"]) == tc["next_item"]
      end)
    end
  end

  test "LIB-2: total items conserved" do
    for sequences <- Enum.take(dataset_gen(), @num_runs) do
      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      orig_total = Enum.sum(Enum.map(sequences, fn s -> length(s["sequence"]) end))
      train_total = Enum.sum(Enum.map(train_seqs, fn s -> length(Convert.seq_from(s)) end))

      assert train_total + length(test_cases) == orig_total
    end
  end

  test "LIB-3: train sequence equals test context for long originals" do
    for sequences <- Enum.take(dataset_gen(), @num_runs) do
      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      # Only long originals get truncated in train; pair by position
      long_pairs =
        Enum.zip(sequences, train_seqs)
        |> Enum.filter(fn {orig, _} -> length(orig["sequence"]) > 2 end)
        |> Enum.map(fn {_, tr} -> tr end)

      assert length(long_pairs) == length(test_cases)

      Enum.zip(long_pairs, test_cases)
      |> Enum.each(fn {tr, tc} ->
        assert Convert.seq_from(tr) == tc["context"]
      end)
    end
  end

  test "LIB-5: timestamps match original prefix for long sequences" do
    for sequences <- Enum.take(dataset_gen(), @num_runs) do
      {:ok, train_seqs, _} = Convert.split_train_test_chrono(sequences, 0.2)

      long_pairs =
        Enum.zip(sequences, train_seqs)
        |> Enum.filter(fn {orig, _} -> length(orig["sequence"]) > 2 end)

      Enum.each(long_pairs, fn {orig, tr} ->
        orig_ts = orig["timestamps"]
        tr_seq = Convert.seq_from(tr)
        tr_ts = Map.get(tr, "timestamps", [])

        if tr_ts != [] do
          expected_ts = Enum.take(orig_ts, length(tr_seq))
          assert tr_ts == expected_ts
        end
      end)
    end
  end
end
