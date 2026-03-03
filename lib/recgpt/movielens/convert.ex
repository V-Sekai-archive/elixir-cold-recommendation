defmodule RecGPT.MovieLens.Convert do
  @moduledoc """
  Converts MovieLens 20M CSV data into RecGPT canonical JSON artifacts.

  Reads ratings.csv and movies.csv, builds sessions (per-user, timestamp-ordered),
  splits 80% train / 20% test (last-item-out), defines cold items (≤ K sessions in train),
  and writes items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json.
  """

  @default_cold_k 2
  @default_train_ratio 0.8
  @max_context 64

  @doc """
  Converts MovieLens data from `src_dir` and writes JSON to `out_dir`.

  Options:
  - `:max_items` - Cap catalog size (default: no cap). Items beyond this are dropped.
  - `:cold_k` - Items in ≤ K train sessions are "cold" (default: 2).
  - `:train_ratio` - Fraction of sessions for train (default: 0.8).

  Returns `:ok` or `{:error, reason}`.
  """
  def run(src_dir, out_dir, opts \\ []) do
    src = Path.expand(src_dir, File.cwd!())
    out = Path.expand(out_dir, File.cwd!())

    unless File.dir?(src) do
      return_error("Source directory not found: #{src}")
    end

    ratings_path = Path.join(src, "ratings.csv")
    movies_path = Path.join(src, "movies.csv")

    unless File.regular?(ratings_path), do: return_error("ratings.csv not found: #{ratings_path}")
    unless File.regular?(movies_path), do: return_error("movies.csv not found: #{movies_path}")

    max_items = opts[:max_items]
    cold_k = opts[:cold_k] || @default_cold_k
    train_ratio = opts[:train_ratio] || @default_train_ratio

    File.mkdir_p!(out)

    with {:ok, movies} <- load_movies(movies_path),
         {:ok, sessions} <- load_sessions(ratings_path),
         {:ok, item_map, items} <- build_catalog(movies, sessions, max_items),
         {:ok, train, test} <- split_sessions(sessions, item_map, train_ratio),
         cold_set <- compute_cold_set(train, cold_k),
         cold_train <- filter_cold_train(train, cold_set),
         cold_test <- filter_cold_test(test, cold_set) do
      num_items = length(items)

      write_items(out, items)
      write_train(out, train, num_items)
      write_test(out, test, num_items)
      write_cold_train(out, cold_train, num_items)
      write_cold_test(out, cold_test, num_items)

      Mix.shell().info("Wrote #{out}/items.json (#{num_items} items)")
      Mix.shell().info("Wrote #{out}/train_sequences.json (#{length(train)} sequences)")
      Mix.shell().info("Wrote #{out}/test_sequences.json (#{length(test)} test cases)")
      Mix.shell().info("Wrote #{out}/cold_train_sequences.json (#{length(cold_train)} sequences)")
      Mix.shell().info("Wrote #{out}/cold_test_sequences.json (#{length(cold_test)} test cases)")

      :ok
    end
  end

  defp return_error(msg), do: {:error, msg}

  defp load_movies(path) do
    content = File.read!(path)
    [_header | rows] = NimbleCSV.RFC4180.parse_string(content)

    # header: movieId,title,genres
    movies =
      Enum.map(rows, fn row ->
        [mid_str, title | _] = row
        {String.to_integer(mid_str), title}
      end)

    {:ok, Map.new(movies)}
  end

  defp load_sessions(path) do
    # ratings: userId,movieId,rating,timestamp — stream and reduce by user to avoid 20M list
    user_events =
      path
      |> File.stream!([:raw, :read_ahead, :binary], :line)
      |> Stream.drop(1)
      |> Stream.map(&String.trim_trailing(&1, "\n"))
      |> Stream.reject(&(String.trim(&1) == ""))
      |> Enum.reduce(%{}, fn line, acc ->
        [uid, mid, _rating, ts] = String.split(line, ",", parts: 4)
        event = {String.to_integer(mid), String.to_integer(ts)}
        Map.update(acc, String.to_integer(uid), [event], &[event | &1])
      end)

    # Per user: sort by timestamp, extract movieIds, keep only sequences with ≥2 items
    sessions =
      user_events
      |> Enum.map(fn {_uid, events} ->
        events
        |> Enum.sort_by(&elem(&1, 1))
        |> Enum.map(&elem(&1, 0))
      end)
      |> Enum.reject(&(length(&1) < 2))

    {:ok, sessions}
  end

  defp build_catalog(movies, sessions, max_items) do
    # Items = movies that appear in ratings, sorted by movieId, remapped to 0-based
    all_movie_ids =
      sessions
      |> Enum.flat_map(& &1)
      |> MapSet.new()

    ordered =
      all_movie_ids
      |> Enum.sort()
      |> then(fn ids ->
        if max_items && max_items > 0, do: Enum.take(ids, max_items), else: ids
      end)

    item_map = Map.new(Enum.with_index(ordered), fn {old, i} -> {old, i} end)
    items = Enum.map(item_map, fn {old_id, new_id} -> %{"id" => new_id, "title" => movies[old_id] || "Movie #{old_id}"} end)

    {:ok, item_map, items}
  end

  defp split_sessions(sessions, item_map, train_ratio) do
    # Only keep sessions where all items are in item_map
    valid =
      Enum.filter(sessions, fn seq ->
        Enum.all?(seq, &Map.has_key?(item_map, &1))
      end)

    n = length(valid)
    train_n = max(1, floor(n * train_ratio))
    {train_raw, test_raw} = Enum.split(valid, train_n)

    train = Enum.map(train_raw, fn seq -> Enum.map(seq, &Map.fetch!(item_map, &1)) end)
    test = Enum.map(test_raw, fn seq -> Enum.map(seq, &Map.fetch!(item_map, &1)) end)

    {:ok, train, test}
  end

  defp compute_cold_set(train, k) do
    # Item -> number of train SESSIONS (sequences) containing it
    counts =
      Enum.reduce(train, %{}, fn seq, acc ->
        for item <- MapSet.new(seq), reduce: acc do
          a -> Map.update(a, item, 1, &(&1 + 1))
        end
      end)

    counts
    |> Enum.filter(fn {_item, cnt} -> cnt <= k end)
    |> MapSet.new(fn {item, _} -> item end)
  end

  defp filter_cold_train(train, cold_set) do
    Enum.filter(train, fn seq -> Enum.any?(seq, &MapSet.member?(cold_set, &1)) end)
  end

  defp filter_cold_test(test, cold_set) do
    test
    |> Enum.map(&seq_to_test_case/1)
    |> Enum.filter(fn tc -> MapSet.member?(cold_set, tc["next_item"]) end)
  end

  defp seq_to_test_case([]), do: %{"context" => [], "next_item" => 0}
  defp seq_to_test_case([single]), do: %{"context" => [], "next_item" => single}
  defp seq_to_test_case(seq) do
    context = seq |> Enum.drop(-1) |> Enum.take(-@max_context)
    next_item = List.last(seq)
    %{"context" => context, "next_item" => next_item}
  end

  defp write_items(out, items) do
    num_items = length(items)
    json = Jason.encode!(%{"items" => items, "num_items" => num_items}, pretty: true)
    tmp = Path.join(out, "items.json.tmp")
    final = Path.join(out, "items.json")
    File.write!(tmp, json)
    File.rename!(tmp, final)
  end

  defp write_train(out, sequences, num_items) do
    json = Jason.encode!(%{"sequences" => sequences, "num_items" => num_items}, pretty: true)
    File.write!(Path.join(out, "train_sequences.json"), json)
  end

  defp write_test(out, test, num_items) do
    test_cases = Enum.map(test, &seq_to_test_case/1)
    json = Jason.encode!(%{"test_cases" => test_cases, "num_items" => num_items}, pretty: true)
    File.write!(Path.join(out, "test_sequences.json"), json)
  end

  defp write_cold_train(out, sequences, num_items) do
    json = Jason.encode!(%{"sequences" => sequences, "num_items" => num_items}, pretty: true)
    File.write!(Path.join(out, "cold_train_sequences.json"), json)
  end

  defp write_cold_test(out, test_cases, num_items) do
    json = Jason.encode!(%{"test_cases" => test_cases, "num_items" => num_items}, pretty: true)
    File.write!(Path.join(out, "cold_test_sequences.json"), json)
  end
end
