defmodule RecGPT.Clickstream.Fetch do
  @moduledoc """
  Download UCI Clickstream Data for Online Shopping and load into SQLite via Ecto.
  Builds data/clickstream/ for eval. 165,474 rows, 217 products, CC BY 4.0.
  """
  import Ecto.Query

  @uci_zip_url "https://archive.ics.uci.edu/static/public/553/clickstream+data+for+online+shopping.zip"
  @csv_name "e-shop clothing 2008.csv"
  @batch_size 10_000
  @source_dataset "uci_clickstream"
  @train_ratio 0.8

  @doc """
  Download zip, extract, run migrations, load into Repo.
  Writes data/clickstream/items.json and test_sequences.json for eval.
  Returns :ok or {:error, reason}.
  """
  def run(data_dir \\ "data/clickstream") do
    data_path = Path.expand(data_dir, File.cwd!())
    zip_path = Path.join(data_path, "clickstream.zip")
    File.mkdir_p!(data_path)

    with :ok <- ensure_zip(zip_path),
         {:ok, csv_path} <- extract_csv(zip_path, data_path),
         :ok <- migrate(),
         {:ok, num_items} <- load_items_and_events(csv_path),
         :ok <- write_eval_artifacts(data_path, num_items) do
      :ok
    end
  end

  defp ensure_zip(zip_path) do
    if File.regular?(zip_path) do
      Mix.shell().info("Using existing #{zip_path}")
      :ok
    else
      Mix.shell().info("Downloading UCI Clickstream (~776 KB)...")
      file = File.open!(zip_path, [:write, :binary, :raw])
      try do
        case Req.get(@uci_zip_url,
               into: fn {:data, data}, acc ->
                 IO.binwrite(file, data)
                 {:cont, acc}
               end
             ) do
          {:ok, %{status: 200}} -> :ok
          {:ok, %{status: s}} -> {:error, "HTTP #{s}"}
          {:error, reason} -> {:error, reason}
        end
      after
        File.close(file)
      end
    end
  end

  defp extract_csv(zip_path, out_dir) do
    Mix.shell().info("Extracting...")
    result = :zip.extract(to_charlist(zip_path), [{:cwd, to_charlist(out_dir)}])
    case result do
      :ok -> find_csv(out_dir)
      {:ok, _paths} -> find_csv(out_dir)
      {:error, reason} -> {:error, reason}
    end
  end

  defp find_csv(out_dir) do
    flat = Path.join(out_dir, @csv_name)
    sub = Path.join(out_dir, "clickstream+data+for+online+shopping/#{@csv_name}")
    cond do
      File.regular?(flat) -> {:ok, flat}
      File.regular?(sub) -> {:ok, sub}
      true -> {:error, "CSV not found after extract"}
    end
  end

  defp migrate do
    Mix.shell().info("Running migrations...")
    path = migrations_path()
    case Ecto.Migrator.run(RecGPT.Repo, path, :up, all: true) do
      list when is_list(list) -> :ok
      other -> other
    end
  end

  defp migrations_path do
    cwd = File.cwd!()
    from_cwd = Path.join([cwd, "priv", "repo", "migrations"])
    if File.exists?(from_cwd), do: from_cwd, else: Path.join(Application.app_dir(:recgpt), "priv/repo/migrations")
  end

  defp load_items_and_events(csv_path) do
    Mix.shell().info("Parsing CSV...")
    {rows, header} = parse_csv(csv_path)
    idx_session = find_index(header, "session ID")
    idx_order = find_index(header, "order")
    idx_page1 = find_index(header, "page 1 (main category)")
    idx_page2 = find_index(header, "page 2 (clothing model)")
    idx_colour = find_index(header, "colour")

    if idx_session == nil or idx_order == nil or idx_page2 == nil do
      raise "Expected columns not found. Header: #{inspect(header)}"
    end

    keys_to_id = %{}
    id_to_title = %{}
    {keys_to_id, id_to_title, _next} =
      Enum.reduce(rows, {keys_to_id, id_to_title, 0}, fn row, {map, titles, next_id} ->
        p1 = safe_at(row, idx_page1)
        p2 = safe_at(row, idx_page2)
        col = safe_at(row, idx_colour)
        key = {p1, p2, col}
        case Map.get(map, key) do
          nil ->
            title = "category #{p1} product #{p2} colour #{col}" |> String.trim()
            map = Map.put(map, key, next_id)
            titles = Map.put(titles, next_id, title)
            {map, titles, next_id + 1}
          _ ->
            {map, titles, next_id}
        end
      end)

    num_items = map_size(id_to_title)
    Mix.shell().info("Loading #{num_items} items, #{length(rows)} events...")

    now = DateTime.utc_now()

    catalog_item_rows =
      for {id, title} <- id_to_title do
        %{
          item_id: id,
          source_dataset: @source_dataset,
          dc_title: title,
          dc_description: title,
          dcterms_source: @source_dataset,
          inserted_at: now,
          updated_at: now
        }
      end

    RecGPT.Repo.insert_all(
      RecGPT.Clickstream.CatalogItem,
      catalog_item_rows,
      on_conflict: :replace_all,
      conflict_target: [:item_id, :source_dataset]
    )

    etnf_events =
      Enum.map(rows, fn row ->
        session_id = safe_at_int(row, idx_session)
        ord = safe_at_int(row, idx_order)
        key = {safe_at(row, idx_page1), safe_at(row, idx_page2), safe_at(row, idx_colour)}
        item_id = Map.fetch!(keys_to_id, key)
        %{
          session_id: session_id,
          ord: ord,
          item_id: item_id,
          source_dataset: @source_dataset,
          inserted_at: now,
          updated_at: now
        }
      end)

    etnf_events
    |> Enum.chunk_every(@batch_size)
    |> Enum.each(fn batch ->
      RecGPT.Repo.insert_all(
        RecGPT.Clickstream.EtnfEvent,
        batch,
        on_conflict: :replace_all,
        conflict_target: [:session_id, :ord]
      )
    end)

    {:ok, num_items}
  end

  defp parse_csv(csv_path) do
    [header_line | rest] =
      csv_path
      |> File.stream!([:read_ahead], 32 * 1024)
      |> Enum.to_list()

    sep = if String.contains?(header_line, ";"), do: ";", else: ","
    header = parse_csv_line(header_line, sep)
    rows = Enum.map(rest, fn line -> parse_csv_line(String.trim_trailing(line, "\n"), sep) end)
    {rows, header}
  end

  defp parse_csv_line(line, sep) do
    if line == "" do
      []
    else
      String.split(line, sep) |> Enum.map(&String.trim(&1, "\""))
    end
  end

  defp find_index(header, name) do
    Enum.find_index(header, fn h -> String.downcase(String.trim(h)) == String.downcase(name) end)
  end

  defp safe_at(_row, nil), do: ""
  defp safe_at(row, i) when is_list(row), do: Enum.at(row, i) || ""

  defp safe_at_int(_row, nil), do: 0
  defp safe_at_int(row, i) when is_list(row) do
    s = Enum.at(row, i) || "0"
    case Integer.parse(to_string(s)) do
      {n, _} -> n
      :error -> 0
    end
  end

  defp write_eval_artifacts(data_path, num_items) do
    query =
      from e in RecGPT.Clickstream.EtnfEvent,
        order_by: [asc: e.session_id, asc: e.ord],
        select: %{session_id: e.session_id, ord: e.ord, item_id: e.item_id}

    events = RecGPT.Repo.all(query)
    sessions =
      events
      |> Enum.group_by(& &1.session_id, & &1)
      |> Enum.filter(fn {_sid, list} -> length(list) >= 2 end)
      |> Enum.sort_by(fn {sid, _} -> sid end)

    {train_sessions, test_sessions} = split_train_test(sessions)

    train_sequences =
      for {_sid, list} <- train_sessions do
        list |> Enum.sort_by(& &1.ord) |> Enum.map(& &1.item_id)
      end

    test_cases = for {_sid, list} <- test_sessions, do: build_test_case(list)

    items =
      RecGPT.Repo.all(RecGPT.Clickstream.CatalogItem)
      |> Enum.filter(&(&1.source_dataset == @source_dataset))
      |> Enum.sort_by(& &1.item_id)
      |> Enum.map(fn i -> %{"id" => i.item_id, "title" => i.dc_title} end)

    items_json = Path.join(data_path, "items.json")
    File.write!(items_json, Jason.encode!(%{"items" => items, "num_items" => num_items}, pretty: true))
    Mix.shell().info("Wrote #{items_json}")

    train_json = Path.join(data_path, "train_sequences.json")
    File.write!(train_json, Jason.encode!(%{"sequences" => train_sequences, "num_items" => num_items}, pretty: true))
    Mix.shell().info("Wrote #{train_json} (#{length(train_sequences)} train sequences)")

    test_json = Path.join(data_path, "test_sequences.json")
    File.write!(test_json, Jason.encode!(%{"test_cases" => test_cases, "num_items" => num_items}, pretty: true))
    Mix.shell().info("Wrote #{test_json} (#{length(test_cases)} test cases)")

    :ok
  end

  defp split_train_test(sessions) do
    n = length(sessions)
    train_n = max(1, round(n * @train_ratio))
    {Enum.take(sessions, train_n), Enum.drop(sessions, train_n)}
  end

  defp build_test_case(list) do
    sorted = Enum.sort_by(list, & &1.ord)
    ids = Enum.map(sorted, & &1.item_id)
    context = ids |> Enum.drop(-1) |> Enum.take(-64)
    next_item = List.last(ids)
    %{"context" => context, "next_item" => next_item}
  end
end
