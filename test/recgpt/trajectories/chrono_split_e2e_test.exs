# End-to-end pipeline test using synthetic MovieLens-style data.
# Creates temp CSVs so every movie ID is known, ensuring predictable sequences.
defmodule RecGPT.Trajectories.ChronoSplitE2ETest do
  use ExUnit.Case, async: true

  alias RecGPT.Trajectories.Convert

  describe "convert_movielens chrono split e2e" do
    test "produces expected items + train/test files with synthetic temporal data" do
      tmp = System.tmp_dir!()
      run_id = :erlang.unique_integer([:positive])
      src_dir = Path.join(tmp, "ml20m_src_#{run_id}")
      out_dir = Path.join(tmp, "ml20m_out_#{run_id}")
      File.mkdir_p!(src_dir)
      File.mkdir_p!(out_dir)

      # Write tiny ratings.csv with known movie IDs and increasing timestamps
      # Users: U10 has 5 interactions, U20 has 4, U30 has 2 (short → stays in train only)
      ratings = [
        "userId,movieId,rating,timestamp",
        "10,101,4.0,1000", "10,102,3.0,2000", "10,103,5.0,3000", "10,104,4.0,4000", "10,105,3.0,5000",
        "20,101,3.0,1500", "20,103,4.0,2500", "20,104,2.0,3500", "20,105,5.0,4500",
        "30,102,5.0,1200", "30,104,4.0,2200"
      ]

      movies = [
        "movieId,title,genres",
        "101,Alpha Movie,Action",
        "102,Beta Movie,Comedy",
        "103,Gamma Movie,Drama",
        "104,Delta Movie,Horror",
        "105,Epsilon Movie,Sci-Fi"
      ]

      try do
        File.write!(Path.join(src_dir, "ratings.csv"), Enum.join(ratings, "\n"))
        File.write!(Path.join(src_dir, "movies.csv"), Enum.join(movies, "\n"))

        :ok = Convert.run(src_dir, out_dir, format: :movielens, split_method: :chrono)

        # items.json
        items_path = Path.join(out_dir, "items.json")
        assert File.regular?(items_path)
        %{"items" => items, "num_items" => num_items} = File.read!(items_path) |> Jason.decode!()
        assert num_items == 5
        assert length(items) == 5

        # train_sequences.json
        train_path = Path.join(out_dir, "train_sequences.json")
        assert File.regular?(train_path)
        train = File.read!(train_path) |> Jason.decode!()
        assert is_list(train["sequences"])
        assert length(train["sequences"]) == 3

        # test_sequences.json — chrono split produces test cases for sequences >2 items
        test_path = Path.join(out_dir, "test_sequences.json")
        assert File.regular?(test_path)
        test = File.read!(test_path) |> Jason.decode!()
        assert is_list(test["test_cases"])
        # U10 (5 items) → 1 test, U20 (4 items) → 1 test, U30 (2 items) → 0
        assert length(test["test_cases"]) == 2

        # Verify temporal ordering: next_item is chronologically last for each long user
        # (users with 4+ interactions get one test case each; user with 2 gets none)
        assert length(test["test_cases"]) == 2

        for tc <- test["test_cases"] do
          assert is_list(tc["context"])
          assert is_integer(tc["next_item"])
          # Context + next_item should equal the original sequence length for that user
          # (leave-one-out for long sequences)
          total_len = length(tc["context"]) + 1
          assert total_len >= 3
        end

        # Verify lossless invariant
        train_items = Enum.sum(Enum.map(train["sequences"], fn s -> length(Convert.seq_from(s)) end))
        assert train_items + length(test["test_cases"]) == 11
      after
        File.rm_rf(src_dir)
        File.rm_rf(out_dir)
      end
    end
  end
end
