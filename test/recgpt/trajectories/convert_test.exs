defmodule RecGPT.Trajectories.ConvertTest do
  use ExUnit.Case, async: true

  @tmp_dir "tmp/movielens_convert_test"
  @out_dir "data/trajectories_convert_test_out"

  setup do
    File.mkdir_p!(@tmp_dir)
    File.mkdir_p!(@out_dir)

    ratings = """
    userId,movieId,rating,timestamp
    1,100,4,1000
    1,101,5,2000
    1,102,3,3000
    2,100,4,1500
    2,103,5,2500
    """

    movies = """
    movieId,title,genres
    100,Movie A,Action
    101,Movie B,Drama
    102,Movie C,Comedy
    103,Movie D,Sci-Fi
    """

    File.write!(Path.join(@tmp_dir, "ratings.csv"), ratings)
    File.write!(Path.join(@tmp_dir, "movies.csv"), movies)

    on_exit(fn ->
      File.rm_rf(@tmp_dir)
      File.rm_rf(@out_dir)
    end)

    :ok
  end

  describe "convert_movielens" do
    test "produces canonical JSON files" do
      assert :ok = RecGPT.Trajectories.Convert.run(@tmp_dir, @out_dir)

      items = File.read!(Path.join(@out_dir, "items.json")) |> Jason.decode!()
      assert items["num_items"] == 4
      assert length(items["items"]) == 4
      assert Enum.at(items["items"], 0)["title"] == "Movie A"

      train = File.read!(Path.join(@out_dir, "train_sequences.json")) |> Jason.decode!()
      assert train["num_items"] == 4
      assert is_list(train["sequences"])
      assert length(train["sequences"]) >= 1

      test_data = File.read!(Path.join(@out_dir, "test_sequences.json")) |> Jason.decode!()
      assert test_data["num_items"] == 4
      assert is_list(test_data["test_cases"])
      assert length(test_data["test_cases"]) >= 1

      tc = hd(test_data["test_cases"])
      assert Map.has_key?(tc, "context")
      assert Map.has_key?(tc, "next_item")
      assert is_list(tc["context"])
      assert is_integer(tc["next_item"])
    end

    test "respects train_limit and test_limit" do
      assert :ok =
               RecGPT.Trajectories.Convert.run(@tmp_dir, @out_dir,
                 train_limit: 1,
                 test_limit: 1
               )

      train = File.read!(Path.join(@out_dir, "train_sequences.json")) |> Jason.decode!()
      assert length(train["sequences"]) <= 1

      test_data = File.read!(Path.join(@out_dir, "test_sequences.json")) |> Jason.decode!()
      assert length(test_data["test_cases"]) <= 1
    end
  end
end
