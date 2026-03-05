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
      assert train["sequences"] != []

      test_data = File.read!(Path.join(@out_dir, "test_sequences.json")) |> Jason.decode!()
      assert test_data["num_items"] == 4
      assert is_list(test_data["test_cases"])
      assert test_data["test_cases"] != []

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

  describe "convert_kuairand" do
    @kuairand_dir "tmp/kuairand_convert_test"
    @kuairand_out "data/kuairand_convert_test_out"

    setup do
      File.mkdir_p!(@kuairand_dir)
      File.mkdir_p!(@kuairand_out)

      log = """
      user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,is_forward,is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,comment_stay_time,is_profile_enter,is_rand,tab
      0,10,20220411,1900,1649675512388,0,0,0,0,0,0,0,1385,209900,0,0,0,0,1
      0,20,20220416,2000,1650111976017,0,0,0,0,0,0,0,0,65400,0,0,0,0,0
      0,10,20220420,1600,1650444367095,0,0,0,0,0,0,0,1405,170833,0,0,0,0,1
      1,20,20220411,1100,1649645295928,0,0,0,0,0,0,0,0,255160,0,0,0,0,8
      1,30,20220411,1100,1649648827559,0,0,0,0,0,0,0,1970,79733,0,0,0,0,1
      """

      videos = """
      video_id,author_id,video_type,upload_dt,upload_type,visible_status,video_duration,server_width,server_height,music_id,music_type,tag
      10,123,NORMAL,2022-04-10,LongImport,0.0,87433.0,720.0,1280.0,9155697141,9.0,39
      20,456,NORMAL,2022-04-10,Kmovie,0.0,218066.0,720.0,1280.0,6355810746,9.0,2
      30,789,NORMAL,2022-04-09,ShortImport,0.0,9233.0,720.0,1280.0,6618412736,4.0,1
      """

      File.write!(Path.join(@kuairand_dir, "log_standard_4_08_to_4_21_pure.csv"), log)
      File.write!(Path.join(@kuairand_dir, "video_features_basic_pure.csv"), videos)

      on_exit(fn ->
        File.rm_rf(@kuairand_dir)
        File.rm_rf(@kuairand_out)
      end)

      :ok
    end

    test "produces canonical JSON files" do
      assert :ok =
               RecGPT.Trajectories.Convert.run(@kuairand_dir, @kuairand_out, format: :kuairand)

      items = File.read!(Path.join(@kuairand_out, "items.json")) |> Jason.decode!()
      assert items["num_items"] == 3
      assert length(items["items"]) == 3

      train = File.read!(Path.join(@kuairand_out, "train_sequences.json")) |> Jason.decode!()
      assert train["num_items"] == 3
      assert is_list(train["sequences"])
      assert train["sequences"] != []

      test_data = File.read!(Path.join(@kuairand_out, "test_sequences.json")) |> Jason.decode!()
      assert test_data["num_items"] == 3
      assert is_list(test_data["test_cases"])

      tc = hd(test_data["test_cases"])
      assert Map.has_key?(tc, "context")
      assert Map.has_key?(tc, "next_item")
    end
  end
end
