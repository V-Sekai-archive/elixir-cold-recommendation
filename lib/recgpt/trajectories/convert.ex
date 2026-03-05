defmodule RecGPT.Trajectories.Convert do
  @moduledoc """
  Converts raw trajectory datasets (MovieLens, KuaiRand, etc.) to RecGPT canonical JSON format.

  Produces: items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json.

  See docs/features/05_eval_data_shapes.md.
  """

  @max_context 64
  @cold_k 2
  @default_train_limit 10_000
  @default_test_limit 2_000

  @doc """
  Converts a dataset to RecGPT canonical format.

  Options:
    * `:format` - `:movielens` (default), `:ml1m` (MovieLens 1M .dat with title+genres), `:kuairand`
    * `:train_limit` - Max train sequences (default: 10_000). Use 0 for no cap.
    * `:test_limit` - Max test cases (default: 2_000). Use 0 for no cap.
    * `:seed` - Random seed for reproducible subset (default: 42)
    * `:sync_to_db` - When true, sync items and sequences to SQLite (ETNF tables).
      Requires RECGPT_SQLITE_PATH. Skips writing sequence JSON files. Always writes items.json.

  Returns `:ok` or `{:error, reason}`.
  """
  @spec run(String.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def run(from_dir, out_dir, opts \\ []) do
    format = Keyword.get(opts, :format, :movielens)
    train_limit = Keyword.get(opts, :train_limit, @default_train_limit)
    test_limit = Keyword.get(opts, :test_limit, @default_test_limit)
    seed = Keyword.get(opts, :seed, 42)
    sync_to_db = Keyword.get(opts, :sync_to_db, false)

    from_dir = Path.expand(from_dir, File.cwd!())
    out_dir = Path.expand(out_dir, File.cwd!())
    File.mkdir_p!(out_dir)

    convert_opts = [
      train_limit: train_limit,
      test_limit: test_limit,
      seed: seed,
      sync_to_db: sync_to_db
    ]

    case format do
      :movielens -> convert_movielens(from_dir, out_dir, convert_opts)
      :ml1m -> convert_ml1m(from_dir, out_dir, convert_opts)
      :kuairand -> convert_kuairand(from_dir, out_dir, convert_opts)
      other -> {:error, "unsupported format: #{inspect(other)}"}
    end
  end

  defp convert_kuairand(from_dir, out_dir, opts) do
    train_limit = Keyword.get(opts, :train_limit, @default_train_limit)
    test_limit = Keyword.get(opts, :test_limit, @default_test_limit)
    seed = Keyword.get(opts, :seed, 42)
    sync_to_db = Keyword.get(opts, :sync_to_db, false)

    log_candidates = [
      "log_standard_4_08_to_4_21_pure.csv",
      "log_standard_4_22_to_5_08_pure.csv",
      "log_random_4_22_to_5_08_pure.csv"
    ]

    log_paths =
      log_candidates
      |> Enum.map(fn name -> Path.join(from_dir, name) end)
      |> Enum.filter(&File.regular?/1)

    videos_path =
      find_file(from_dir, [
        "video_features_basic_pure.csv",
        "KuaiRand-Pure/video_features_basic_pure.csv"
      ])

    if log_paths == [] do
      {:error,
       "no KuaiRand log CSV found in #{from_dir} (expected log_standard_*.csv or log_random_*.csv)"}
    else
      with {:ok, interactions} <- parse_kuairand_logs(log_paths),
           {:ok, titles} <- parse_kuairand_videos(videos_path),
           {:ok, item_ids, old_to_new} <- build_item_map_kuairand(interactions, titles),
           {:ok, sequences} <- build_sequences_kuairand(interactions, old_to_new),
           {:ok, train_seqs, test_cases} <- split_train_test(sequences, seed),
           train_seqs <- maybe_take(train_seqs, train_limit),
           test_cases <- maybe_take(test_cases, test_limit),
           cold_set <- cold_items(train_seqs, @cold_k),
           cold_test <- Enum.filter(test_cases, fn tc -> tc["next_item"] in cold_set end),
           cold_train <- cold_train_sequences(train_seqs, cold_set) do
        num_items = map_size(old_to_new)

        items =
          Enum.map(Enum.with_index(item_ids), fn {old_id, i} ->
            %{"id" => i, "title" => Map.get(titles, old_id, "")}
          end)

        emit_output(
          out_dir,
          items,
          train_seqs,
          test_cases,
          cold_test,
          cold_train,
          num_items,
          sync_to_db
        )

        :ok
      end
    end
  end

  defp convert_movielens(from_dir, out_dir, opts) do
    train_limit = Keyword.get(opts, :train_limit, @default_train_limit)
    test_limit = Keyword.get(opts, :test_limit, @default_test_limit)
    seed = Keyword.get(opts, :seed, 42)
    sync_to_db = Keyword.get(opts, :sync_to_db, false)

    ratings_path =
      find_file(from_dir, ["ratings.csv", "ml-20m/ratings.csv", "movielens-20m/ratings.csv"])

    movies_path =
      find_file(from_dir, ["movies.csv", "ml-20m/movies.csv", "movielens-20m/movies.csv"])

    unless ratings_path do
      {:error, "ratings.csv not found in #{from_dir}"}
    end

    unless movies_path do
      {:error, "movies.csv not found in #{from_dir}"}
    end

    with {:ok, ratings} <- parse_movielens_ratings(ratings_path),
         {:ok, titles} <- parse_movielens_movies(movies_path),
         {:ok, item_ids, old_to_new} <- build_item_map(ratings, titles),
         {:ok, sequences} <- build_sequences(ratings, old_to_new),
         {:ok, train_seqs, test_cases} <- split_train_test(sequences, seed),
         train_seqs <- maybe_take(train_seqs, train_limit),
         test_cases <- maybe_take(test_cases, test_limit),
         cold_set <- cold_items(train_seqs, @cold_k),
         cold_test <- Enum.filter(test_cases, fn tc -> tc["next_item"] in cold_set end),
         cold_train <- cold_train_sequences(train_seqs, cold_set) do
      num_items = map_size(old_to_new)

      items =
        Enum.map(Enum.with_index(item_ids), fn {old_id, i} ->
          %{"id" => i, "title" => Map.get(titles, old_id, "")}
        end)

      emit_output(
        out_dir,
        items,
        train_seqs,
        test_cases,
        cold_test,
        cold_train,
        num_items,
        sync_to_db
      )

      :ok
    end
  end

  # MovieLens 1M: .dat files (:: separator). ratings.dat UserID::MovieID::Rating::Timestamp;
  # movies.dat MovieID::Title::Genres. Joins title + genres so item descriptions have categories.
  defp convert_ml1m(from_dir, out_dir, opts) do
    train_limit = Keyword.get(opts, :train_limit, @default_train_limit)
    test_limit = Keyword.get(opts, :test_limit, @default_test_limit)
    seed = Keyword.get(opts, :seed, 42)
    sync_to_db = Keyword.get(opts, :sync_to_db, false)

    ratings_path =
      find_file(from_dir, ["ml-1m/ratings.dat", "ratings.dat"])

    movies_path =
      find_file(from_dir, ["ml-1m/movies.dat", "movies.dat"])

    unless ratings_path do
      {:error, "ratings.dat not found in #{from_dir} (expected ml-1m/ratings.dat or ratings.dat)"}
    end

    unless movies_path do
      {:error, "movies.dat not found in #{from_dir} (expected ml-1m/movies.dat or movies.dat)"}
    end

    with {:ok, ratings} <- parse_ml1m_ratings(ratings_path),
         {:ok, item_descriptions} <- parse_ml1m_movies(movies_path),
         {:ok, item_ids, old_to_new} <- build_item_map(ratings, item_descriptions),
         {:ok, sequences} <- build_sequences(ratings, old_to_new),
         {:ok, train_seqs, test_cases} <- split_train_test(sequences, seed),
         train_seqs <- maybe_take(train_seqs, train_limit),
         test_cases <- maybe_take(test_cases, test_limit),
         cold_set <- cold_items(train_seqs, @cold_k),
         cold_test <- Enum.filter(test_cases, fn tc -> tc["next_item"] in cold_set end),
         cold_train <- cold_train_sequences(train_seqs, cold_set) do
      num_items = map_size(old_to_new)

      items =
        Enum.map(Enum.with_index(item_ids), fn {old_id, i} ->
          # Title + genres from README so categories are filled
          title = Map.get(item_descriptions, old_id, "")
          %{"id" => i, "title" => title}
        end)

      emit_output(
        out_dir,
        items,
        train_seqs,
        test_cases,
        cold_test,
        cold_train,
        num_items,
        sync_to_db
      )

      :ok
    end
  end

  defp emit_output(out_dir, items, train_seqs, test_cases, cold_test, cold_train, num_items, true) do
    if System.get_env("RECGPT_SQLITE_PATH") in [nil, ""] do
      raise "sync_to_db requires RECGPT_SQLITE_PATH to be set"
    end

    Application.ensure_all_started(:recgpt)
    alias RecGPT.Catalog.Sync

    Sync.sync_items_from_list(items)
    Sync.sync_sequences_from_list(train_seqs, cold_train)
    Sync.sync_test_from_list(test_cases, cold_test)

    n = num_items || length(items)
    File.mkdir_p!(out_dir)
    write_items_json_from_list(out_dir, items, n)
    write_test_json(out_dir, test_cases, "test_sequences.json", n)
    write_test_json(out_dir, cold_test, "cold_test_sequences.json", n)

    require Logger

    Logger.info(
      "Synced items and sequences to SQLite. Use --items db for build_fixture and pretrain."
    )
  end

  defp emit_output(
         out_dir,
         items,
         train_seqs,
         test_cases,
         cold_test,
         cold_train,
         num_items,
         false
       ) do
    write_items_json_from_list(out_dir, items, num_items)
    write_sequences_json(out_dir, train_seqs, "train_sequences.json", "sequences", num_items)
    write_test_json(out_dir, test_cases, "test_sequences.json", num_items)
    write_test_json(out_dir, cold_test, "cold_test_sequences.json", num_items)
    write_sequences_json(out_dir, cold_train, "cold_train_sequences.json", "sequences", num_items)
  end

  defp write_items_json_from_list(out_dir, items, num_items) do
    path = Path.join(out_dir, "items.json")
    File.write!(path, Jason.encode!(%{"items" => items, "num_items" => num_items}, pretty: true))
    require Logger
    Logger.info("Wrote #{path} (#{num_items} items)")
    :ok
  end

  defp maybe_take(list, 0), do: list
  defp maybe_take(list, limit) when limit > 0, do: Enum.take(list, limit)

  defp find_file(dir, candidates) do
    Enum.find(candidates, fn name ->
      path = Path.join(dir, name)
      File.regular?(path)
    end)
    |> case do
      nil -> nil
      name -> Path.join(dir, name)
    end
  end

  defp parse_movielens_ratings(path) do
    content = File.read!(path)
    rows = parse_csv(content)
    [header | data] = rows

    col = fn row, name ->
      idx = Enum.find_index(header, &(String.downcase(&1) == name))
      if idx, do: Enum.at(row, idx), else: nil
    end

    parsed =
      Enum.map(data, fn row ->
        user_id = parse_int(col.(row, "userid"))
        movie_id = parse_int(col.(row, "movieid"))
        timestamp = parse_int(col.(row, "timestamp"))
        {user_id, movie_id, timestamp}
      end)
      |> Enum.filter(fn
        {nil, _, _} -> false
        {_, nil, _} -> false
        {_, _, nil} -> false
        _ -> true
      end)

    {:ok, parsed}
  rescue
    e -> {:error, e}
  end

  # movies.csv: movieId, title, genres (pipe-separated). Join title + genres so categories are filled.
  defp parse_movielens_movies(path) do
    content = File.read!(path)
    rows = parse_csv(content)
    [header | data] = rows

    col = fn row, name ->
      idx = Enum.find_index(header, &(String.downcase(&1) == name))
      if idx, do: Enum.at(row, idx), else: nil
    end

    titles =
      Enum.reduce(data, %{}, fn row, acc ->
        movie_id = parse_int(col.(row, "movieid"))
        title = col.(row, "title") || ""
        genres = col.(row, "genres") || ""
        genres_clean = String.replace(genres, "|", ", ")
        desc = if genres_clean == "", do: title, else: "#{title} | #{genres_clean}"
        if movie_id, do: Map.put(acc, movie_id, desc), else: acc
      end)

    {:ok, titles}
  rescue
    e -> {:error, e}
  end

  defp parse_kuairand_logs(log_paths) do
    parsed =
      Enum.flat_map(log_paths, fn path ->
        content = File.read!(path)
        rows = parse_csv(content)
        [header | data] = rows

        col = fn row, name ->
          idx = Enum.find_index(header, &(String.downcase(&1) == name))
          if idx, do: Enum.at(row, idx), else: nil
        end

        Enum.map(data, fn row ->
          user_id = parse_int(col.(row, "user_id"))
          video_id = parse_int(col.(row, "video_id"))
          time_ms = parse_int(col.(row, "time_ms"))
          {user_id, video_id, time_ms}
        end)
        |> Enum.filter(fn
          {nil, _, _} -> false
          {_, nil, _} -> false
          {_, _, nil} -> false
          _ -> true
        end)
      end)

    {:ok, parsed}
  rescue
    e -> {:error, e}
  end

  defp parse_kuairand_videos(nil), do: {:ok, %{}}

  defp parse_kuairand_videos(path) do
    content = File.read!(path)
    rows = parse_csv(content)
    [header | data] = rows

    col = fn row, name ->
      idx = Enum.find_index(header, &(String.downcase(&1) == name))
      if idx, do: Enum.at(row, idx), else: nil
    end

    titles =
      Enum.reduce(data, %{}, fn row, acc ->
        video_id = parse_int(col.(row, "video_id"))
        tag = col.(row, "tag") || ""
        video_type = col.(row, "video_type") || ""

        title =
          if tag != "" or video_type != "",
            do: "video #{video_id} #{tag} #{video_type}" |> String.trim(),
            else: "video #{video_id}"

        if video_id, do: Map.put(acc, video_id, title), else: acc
      end)

    {:ok, titles}
  rescue
    e -> {:error, e}
  end

  defp build_item_map_kuairand(interactions, titles) do
    item_ids = interactions |> Enum.map(fn {_, v, _} -> v end) |> Enum.uniq()
    all_ids = MapSet.new(item_ids) |> MapSet.union(MapSet.new(Map.keys(titles)))
    sorted = Enum.sort(all_ids)
    old_to_new = Map.new(Enum.with_index(sorted), fn {old, i} -> {old, i} end)
    {:ok, sorted, old_to_new}
  end

  defp build_sequences_kuairand(interactions, old_to_new) do
    by_user = Enum.group_by(interactions, fn {u, _, _} -> u end)

    sequences =
      by_user
      |> Enum.map(fn {_user, rows} ->
        sorted = Enum.sort_by(rows, fn {_, _, t} -> t end)
        ids = Enum.map(sorted, fn {_, video_id, _} -> Map.get(old_to_new, video_id) end) |> Enum.reject(&is_nil/1)
        time_ms = Enum.map(sorted, fn {_, _, t} -> t || 0 end)
        %{"sequence" => ids, "timestamps" => time_ms}
      end)
      |> Enum.filter(fn m -> length(m["sequence"]) >= 2 end)

    {:ok, sequences}
  end

  defp parse_csv(content) do
    NimbleCSV.RFC4180.parse_string(content, skip_headers: false)
  end

  # MovieLens 1M .dat: lines with "::" separator, no header
  defp parse_dat_line(line) do
    line
    |> String.trim()
    |> String.split("::", parts: :infinity)
    |> Enum.map(&String.trim/1)
  end

  # ratings.dat: UserID::MovieID::Rating::Timestamp (seconds since epoch)
  defp parse_ml1m_ratings(path) do
    lines =
      path
      |> File.read!()
      |> String.split(~r/\r?\n/, trim: true)

    parsed =
      Enum.reduce(lines, [], fn line, acc ->
        parts = parse_dat_line(line)
        if length(parts) >= 4 do
          [user_id, movie_id, _rating, ts] = Enum.take(parts, 4)
          user = parse_int(user_id)
          movie = parse_int(movie_id)
          timestamp = parse_int(ts)
          if user && movie && timestamp, do: [{user, movie, timestamp} | acc], else: acc
        else
          acc
        end
      end)
      |> Enum.reverse()

    {:ok, parsed}
  rescue
    e -> {:error, e}
  end

  # movies.dat: MovieID::Title::Genres (pipe-separated). Joins so item title has categories.
  # README: Title from IMDB, Genres pipe-separated (Action, Comedy, etc.)
  defp parse_ml1m_movies(path) do
    content = File.read!(path)
    content = ensure_utf8(content)

    lines = String.split(content, ~r/\r?\n/, trim: true)

    map =
      Enum.reduce(lines, %{}, fn line, acc ->
        parts = parse_dat_line(line)
        if length(parts) >= 3 do
          [movie_id, title, genres] = Enum.take(parts, 3)
          id = parse_int(movie_id)
          title = title || ""
          genres = (genres || "") |> String.replace("|", ", ")
          desc = if genres == "", do: title, else: "#{title} | #{genres}"
          if id, do: Map.put(acc, id, desc), else: acc
        else
          acc
        end
      end)

    {:ok, map}
  rescue
    e -> {:error, e}
  end

  defp ensure_utf8(binary) when is_binary(binary) do
    if String.valid?(binary) do
      binary
    else
      result = :unicode.characters_to_binary(:binary.bin_to_list(binary), :latin1, :utf8)
      if is_tuple(result), do: elem(result, 0), else: result
    end
  end

  defp parse_int(nil), do: nil

  defp parse_int(s) when is_binary(s) do
    case Integer.parse(String.trim(s)) do
      {n, _} -> n
      :error -> nil
    end
  end

  defp parse_int(n) when is_integer(n), do: n
  defp parse_int(_), do: nil

  defp build_item_map(ratings, titles) do
    item_ids = ratings |> Enum.map(fn {_, m, _} -> m end) |> Enum.uniq()
    all_ids = MapSet.new(item_ids) |> MapSet.union(MapSet.new(Map.keys(titles)))
    sorted = Enum.sort(all_ids)
    old_to_new = Map.new(Enum.with_index(sorted), fn {old, i} -> {old, i} end)
    {:ok, sorted, old_to_new}
  end

  defp build_sequences(ratings, old_to_new) do
    by_user = Enum.group_by(ratings, fn {u, _, _} -> u end)

    sequences =
      by_user
      |> Enum.map(fn {_user, rows} ->
        sorted = Enum.sort_by(rows, fn {_, _, t} -> t end)
        ids = Enum.map(sorted, fn {_, movie_id, _} -> Map.get(old_to_new, movie_id) end) |> Enum.reject(&is_nil/1)
        # FuXi Linear: per-position timestamps in ms (Training.build_train_batch does relative-from-start + 4x expand)
        time_ms = Enum.map(sorted, fn {_, _, t} -> (t || 0) * 1000 end)
        %{"sequence" => ids, "timestamps" => time_ms}
      end)
      |> Enum.filter(fn m -> length(m["sequence"]) >= 2 end)

    {:ok, sequences}
  end

  defp seq_from(s) when is_list(s), do: s
  defp seq_from(%{"sequence" => s}) when is_list(s), do: s
  defp seq_from(_), do: []

  defp split_train_test(sequences, seed) do
    :rand.seed(:exs1024, {seed, 0, 0})
    capped = Enum.take(sequences, 100_000)
    shuffled = Enum.shuffle(capped)
    n = length(shuffled)
    split = max(1, min(n - 1, round(n * 0.8)))
    train = Enum.take(shuffled, split)
    test_raw = Enum.drop(shuffled, split)

    test_cases =
      Enum.map(test_raw, fn seq ->
        seq_to_test_case(seq_from(seq))
      end)

    {:ok, train, test_cases}
  end

  defp seq_to_test_case([]), do: %{"context" => [], "next_item" => 0}
  defp seq_to_test_case([single]), do: %{"context" => [], "next_item" => single}

  defp seq_to_test_case(seq) do
    context = seq |> Enum.drop(-1) |> Enum.take(-@max_context)
    next_item = List.last(seq)
    %{"context" => context, "next_item" => next_item}
  end

  defp cold_items(train_seqs, k) do
    counts =
      Enum.reduce(train_seqs, %{}, fn seq, acc ->
        Enum.reduce(Enum.uniq(seq_from(seq)), acc, fn item_id, a ->
          Map.update(a, item_id, 1, &(&1 + 1))
        end)
      end)

    counts
    |> Enum.filter(fn {_, c} -> c <= k end)
    |> Enum.map(fn {id, _} -> id end)
    |> MapSet.new()
  end

  defp cold_train_sequences(train_seqs, cold_set) do
    Enum.filter(train_seqs, fn seq ->
      Enum.any?(seq_from(seq), &MapSet.member?(cold_set, &1))
    end)
  end

  defp write_sequences_json(out_dir, sequences, filename, key, num_items) do
    path = Path.join(out_dir, filename)
    payload = %{key => sequences, "num_items" => num_items}
    File.write!(path, Jason.encode!(payload, pretty: true))
    require Logger
    Logger.info("Wrote #{path} (#{length(sequences)} sequences)")
    :ok
  end

  defp write_test_json(out_dir, test_cases, filename, num_items) do
    path = Path.join(out_dir, filename)
    payload = %{"test_cases" => test_cases, "num_items" => num_items}
    File.write!(path, Jason.encode!(payload, pretty: true))
    require Logger
    Logger.info("Wrote #{path} (#{length(test_cases)} test cases)")
    :ok
  end
end
