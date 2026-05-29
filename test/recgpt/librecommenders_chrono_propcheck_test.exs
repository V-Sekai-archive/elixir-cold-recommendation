defmodule RecGPT.LibrecommendersChronoPropCheckTest do
  @moduledoc """
  PropCheck property tests for temporal chrono split using
  Librecommenders-style sequential data patterns.

  Models the tutorial dataset structure: user-item-rating-time rows
  grouped by user into interaction sequences, then split chronologically
  for next-item recommendation evaluation.

  This mirrors https://librecommender.readthedocs.io/en/latest/tutorial.html
  """
  use ExUnit.Case, async: true
  use PropCheck
  alias RecGPT.Trajectories.Convert

  # ------------------------------------------------------------------
  # Generators
  # ------------------------------------------------------------------

  # Generate a single user's interaction sequence with monotonic timestamps
  @doc """
  Generator for one user's sorted interaction rows.
  Ensures unique items, monotonically increasing timestamps.
  """
  def user_sequence_gen(min_items \\ 2, max_items \\ 20) do
    sized(fn size ->
      count = min(max_items, max(min_items, rem(size, max_items) + min_items))

      # Generate a base timestamp and unique items
      base_ts_gen = pos_integer()
      item_list_gen = fixed_length_list(pos_integer(), count)

      {base_ts_gen, item_list_gen}
      |> bind(fn {base_ts, items} ->
        # Deduplicate items
        uniq = items |> Enum.uniq() |> Enum.take(count)
        # Pad if needed
        uniq =
          if length(uniq) < count do
            max_item = if uniq == [], do: 0, else: Enum.max(uniq)
            extras = for i <- 1..(count - length(uniq)), do: max_item + i
            uniq ++ extras
          else
            uniq
          end

        tss = Enum.map(0..(count - 1), fn i -> base_ts + i * 1000 end)
        ratings = for _ <- 1..count, do: :rand.uniform() * 4.0 + 1.0
        uid = :erlang.unique_integer([:positive])

        rows = Enum.zip([uniq, ratings, tss])

        exactly(
          for {iid, rating, ts} <- rows do
            %{
              "user" => uid,
              "item" => iid,
              "rating" => Float.round(rating, 2),
              "time" => ts
            }
          end
        )
      end)
    end)
  end

  @doc """
  Generator for a list of exactly N elements (workaround for PropCheck list_of min_length).
  """
  def fixed_length_list(gen, n) when n > 0 do
    tuple(List.duplicate(gen, n))
    |> bind(fn t -> exactly(Tuple.to_list(t)) end)
  end

  @doc """
  Dataset generator: multiple user sequences (Librecommenders-style).
  """
  def librecommenders_dataset_gen(min_users \\ 2, max_users \\ 10) do
    integer(min_users, max_users)
    |> bind(fn num_users ->
      # Generate N user sequences independently
      build_list(num_users, &user_sequence_gen/0)
    end)
  end

  @doc """
  Build a list of N elements from a generator.
  """
  def build_list(0, _gen_fn), do: exactly([])

  def build_list(n, gen_fn) when n > 0 do
    gen_fn.()
    |> bind(fn head ->
      build_list(n - 1, gen_fn)
      |> bind(fn tail -> exactly([head | tail]) end)
    end)
  end

  @doc """
  Convert flat interaction rows grouped by user → RecGPT canonical sequences.
  """
  def rows_to_sequences(rows_by_user) do
    rows_by_user
    |> Enum.map(fn rows ->
      sorted = Enum.sort_by(rows, fn r -> r["time"] end)
      ids = Enum.map(sorted, fn r -> r["item"] end)
      tss = Enum.map(sorted, fn r -> r["time"] end)
      %{"sequence" => ids, "timestamps" => tss}
    end)
    |> Enum.filter(fn s -> length(s["sequence"]) >= 1 end)
  end

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  property "LIB-1: chrono split preserves last item as next_item for long sequences" do
    forall dataset <- librecommenders_dataset_gen(2, 8) do
      sequences = rows_to_sequences(dataset)
      {:ok, _train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      long_seqs = Enum.filter(sequences, fn s -> length(s["sequence"]) > 2 end)

      checks =
        if length(long_seqs) == length(test_cases) do
          Enum.all?(Enum.zip(long_seqs, test_cases), fn {orig, tc} ->
            List.last(orig["sequence"]) == tc["next_item"]
          end)
        else
          false
        end

      (length(long_seqs) == length(test_cases)) and checks
    end
  end

  property "LIB-2: total item count is conserved across train and test" do
    forall dataset <- librecommenders_dataset_gen(2, 8) do
      sequences = rows_to_sequences(dataset)
      orig_total = Enum.sum(Enum.map(sequences, fn s -> length(s["sequence"]) end))

      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      train_total = Enum.sum(Enum.map(train_seqs, fn s -> length(Convert.seq_from(s)) end))
      train_total + length(test_cases) == orig_total
    end
  end

  property "LIB-3: short sequences (<=2 items) never produce test cases" do
    forall dataset <- librecommenders_dataset_gen(1, 5) do
      sequences = rows_to_sequences(dataset)

      cond do
        # When all sequences are short, no test cases should be produced
        Enum.all?(sequences, fn s -> length(s["sequence"]) <= 2 end) ->
          {:ok, _train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)
          length(test_cases) == 0

        # When mixed, test_cases only come from long sequences
        true ->
          {:ok, _train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)
          long_count = Enum.count(sequences, fn s -> length(s["sequence"]) > 2 end)
          length(test_cases) == long_count
      end
    end
  end

  property "LIB-4: timestamps match original prefix after train truncation" do
    forall dataset <- librecommenders_dataset_gen(2, 8) do
      sequences = rows_to_sequences(dataset)
      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      long_pairs =
        Enum.zip(sequences, train_seqs)
        |> Enum.filter(fn {orig, _} -> length(orig["sequence"]) > 2 end)

      # For each long sequence, train timestamps match the original's truncated prefix
      Enum.all?(long_pairs, fn {orig, tr} ->
        orig_ts = orig["timestamps"]
        tr_seq = Convert.seq_from(tr)
        tr_ts = Map.get(tr, "timestamps", [])

        if tr_ts != [] and length(orig_ts) >= length(tr_seq) do
          expected_ts = Enum.take(orig_ts, length(tr_seq))
          tr_ts == expected_ts
        else
          true
        end
      end)
    end
  end

  property "LIB-5: test context equals train sequence for matched pairs" do
    forall dataset <- librecommenders_dataset_gen(2, 8) do
      sequences = rows_to_sequences(dataset)
      {:ok, train_seqs, test_cases} = Convert.split_train_test_chrono(sequences, 0.2)

      long_trains =
        Enum.zip(sequences, train_seqs)
        |> Enum.filter(fn {orig, _} -> length(orig["sequence"]) > 2 end)
        |> Enum.map(fn {_, tr} -> tr end)

      if length(long_trains) == length(test_cases) do
        Enum.all?(Enum.zip(long_trains, test_cases), fn {tr, tc} ->
          Convert.seq_from(tr) == tc["context"]
        end)
      else
        false
      end
    end
  end
end
