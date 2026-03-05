defmodule RecGPT.Trajectories.Convert do
  @moduledoc """
  Converts raw trajectory datasets (MovieLens, KuaiRand, etc.) to RecGPT canonical JSON format.

  Produces: items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json.

  See docs/05_eval_data_shapes.md and docs/86_training_signal_test_dataset_plan.md.
  """

  @max_context 64
  @cold_k 2
  @default_train_limit 10_000
  @default_test_limit 2_000

  @doc """
  Converts a dataset to RecGPT canonical format.

  Options:
    * `:format` - `:movielens` (default), `:kuairand` (future)
    * `:train_limit` - Max train sequences (default: 10_000). Use 0 for no cap.
    * `:test_limit` - Max test cases (default: 2_000). Use 0 for no cap.
    * `:seed` - Random seed for reproducible subset (default: 42)

  Returns `:ok` or `{:error, reason}`.
  """
  @spec run(String.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def run(from_dir, out_dir, opts \\ []) do
    format = Keyword.get(opts, :format, :movielens)
    train_limit = Keyword.get(opts, :train_limit, @default_train_limit)
    test_limit = Keyword.get(opts, :test_limit, @default_test_limit)
    seed = Keyword.get(opts, :seed, 42)

    from_dir = Path.expand(from_dir, File.cwd!())
    out_dir = Path.expand(out_dir, File.cwd!())
    File.mkdir_p!(out_dir)

    case format do
      :movielens -> convert_movielens(from_dir, out_dir, train_limit, test_limit, seed)
      other -> {:error, "unsupported format: #{inspect(other)}"}
    end
  end

  defp convert_movielens(from_dir, out_dir, train_limit, test_limit, seed) do
    ratings_path = find_file(from_dir, ["ratings.csv", "ml-20m/ratings.csv", "movielens-20m/ratings.csv"])
    movies_path = find_file(from_dir, ["movies.csv", "ml-20m/movies.csv", "movielens-20m/movies.csv"])

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
      write_items_json(out_dir, item_ids, titles)
      write_sequences_json(out_dir, train_seqs, "train_sequences.json", "sequences", num_items)
      write_test_json(out_dir, test_cases, "test_sequences.json", num_items)
      write_test_json(out_dir, cold_test, "cold_test_sequences.json", num_items)
      write_sequences_json(out_dir, cold_train, "cold_train_sequences.json", "sequences", num_items)
      :ok
    end
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
      idx = Enum.find_index(header, &String.downcase(&1) == name)
      if idx, do: Enum.at(row, idx), else: nil
    end

    parsed =
      Enum.map(data, fn row ->
        userId = parse_int(col.(row, "userid"))
        movieId = parse_int(col.(row, "movieid"))
        timestamp = parse_int(col.(row, "timestamp"))
        {userId, movieId, timestamp}
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

  defp parse_movielens_movies(path) do
    content = File.read!(path)
    rows = parse_csv(content)
    [header | data] = rows

    col = fn row, name ->
      idx = Enum.find_index(header, &String.downcase(&1) == name)
      if idx, do: Enum.at(row, idx), else: nil
    end

    titles =
      Enum.reduce(data, %{}, fn row, acc ->
        movieId = parse_int(col.(row, "movieid"))
        title = col.(row, "title") || ""
        if movieId, do: Map.put(acc, movieId, title), else: acc
      end)

    {:ok, titles}
  rescue
    e -> {:error, e}
  end

  defp parse_csv(content) do
    NimbleCSV.RFC4180.parse_string(content, skip_headers: false)
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
        rows
        |> Enum.sort_by(fn {_, _, t} -> t end)
        |> Enum.map(fn {_, movie_id, _} -> Map.get(old_to_new, movie_id) end)
        |> Enum.reject(&is_nil/1)
      end)
      |> Enum.filter(fn seq -> length(seq) >= 2 end)

    {:ok, sequences}
  end

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
        seq_to_test_case(seq)
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
        Enum.reduce(Enum.uniq(seq), acc, fn item_id, a ->
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
      Enum.any?(seq, &MapSet.member?(cold_set, &1))
    end)
  end

  defp write_items_json(out_dir, item_ids, titles) do
    items =
      Enum.map(Enum.with_index(item_ids), fn {old_id, i} ->
        %{"id" => i, "title" => Map.get(titles, old_id, "")}
      end)

    num_items = length(items)
    path = Path.join(out_dir, "items.json")
    File.write!(path, Jason.encode!(%{"items" => items, "num_items" => num_items}, pretty: true))
    require Logger
    Logger.info("Wrote #{path} (#{num_items} items)")
    :ok
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
