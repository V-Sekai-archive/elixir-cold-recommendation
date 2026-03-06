defmodule RecGPT.Catalog.Sync do
  @moduledoc """
  Flush catalog and sequence data to SQLite (ETNF tables).
  Used by FixtureBuild when RECGPT_SQLITE_PATH is set; flush as soon as each batch is ready.
  """

  alias RecGPT.Catalog.ColdTestCase
  alias RecGPT.Catalog.ColdTestContext
  alias RecGPT.Catalog.ColdTrainSequenceRow
  alias RecGPT.Catalog.{Item, ItemEmbedding, ItemEmbeddingText, ItemToken}
  alias RecGPT.Catalog.TestCase
  alias RecGPT.Catalog.TestContext
  alias RecGPT.Catalog.TrainSequenceRow
  alias RecGPT.Repo

  @doc "Clear items, item_embeddings, item_tokens for a full rebuild."
  def clear_catalog_tables do
    Repo.delete_all(ItemToken)
    Repo.delete_all(ItemEmbedding)
    Repo.delete_all(Item)
  end

  @doc "Insert items batch: list of %{item_id: id, title: title}."
  def insert_items(entries) when is_list(entries) and entries != [] do
    Repo.insert_all(Item, entries)
  end

  @doc "Insert item_embeddings batch: list of %{item_id: id, embedding: binary}."
  def insert_item_embeddings(entries) when is_list(entries) and entries != [] do
    Repo.insert_all(ItemEmbedding, entries)
  end

  @doc "Insert item_embedding_texts batch: list of %{item_id: id, embedding_text: text}."
  def insert_item_embedding_texts(entries) when is_list(entries) and entries != [] do
    Repo.insert_all(ItemEmbeddingText, entries)
  end

  @doc "Insert item_tokens batch: list of %{item_id: id, t0: n, t1: n, t2: n, t3: n}."
  def insert_item_tokens(entries) when is_list(entries) and entries != [] do
    Repo.insert_all(ItemToken, entries)
  end

  @insert_chunk 1000

  @doc """
  Sync items from list to DB. Clears catalog tables first.
  Items: list of %{"id" => id, "title" => title} or %{"id" => id, "title" => title, "embedding_text" => text}.
  When embedding_text is present, syncs to item_embedding_texts as well.
  Enforces ETNF: no duplicate item_ids.
  """
  def sync_items_from_list(items) when is_list(items) do
    clear_catalog_tables()
    Repo.delete_all(ItemEmbeddingText)

    if items != [] do
      entries =
        items
        |> Enum.map(fn %{"id" => id, "title" => title} = item ->
          base = %{item_id: id, title: to_string(title)}
          embed = Map.get(item, "embedding_text")

          embed_entry =
            if embed && String.trim(to_string(embed)) != "" do
              %{item_id: id, embedding_text: to_string(embed)}
            else
              nil
            end

          {base, embed_entry}
        end)
        |> Enum.uniq_by(fn {e, _} -> e.item_id end)

      item_entries = Enum.map(entries, fn {e, _} -> e end)
      item_entries |> Enum.chunk_every(@insert_chunk) |> Enum.each(&insert_items/1)

      embed_entries = entries |> Enum.flat_map(fn {_, e} -> if e, do: [e], else: [] end)

      if embed_entries != [] do
        embed_entries
        |> Enum.chunk_every(@insert_chunk)
        |> Enum.each(&insert_item_embedding_texts/1)
      end
    end

    :ok
  end

  @doc """
  Sync train and cold_train sequences from lists. Clears first.
  Enforces ETNF: deduplicates by (seq_id, pos) before insert.
  """
  def sync_sequences_from_list(train_sequences, cold_train_sequences) do
    Repo.delete_all(ColdTrainSequenceRow)
    Repo.delete_all(TrainSequenceRow)

    if train_sequences != [] do
      rows =
        train_sequences
        |> parse_sequence_rows_from_list()
        |> Enum.uniq_by(fn r -> {r.seq_id, r.pos} end)

      insert_all_in_chunks(TrainSequenceRow, rows)
    end

    if cold_train_sequences != [] do
      rows =
        cold_train_sequences
        |> parse_sequence_rows_from_list()
        |> Enum.uniq_by(fn r -> {r.seq_id, r.pos} end)

      insert_all_in_chunks(ColdTrainSequenceRow, rows)
    end

    :ok
  end

  @doc """
  Sync test_cases and cold_test_cases from lists. Each element: %{"context" => [...], "next_item" => id}.
  Clears first. Enforces ETNF: deduplicates by (case_id, pos) for context rows.
  """
  def sync_test_from_list(test_cases, cold_test_cases) do
    Repo.delete_all(ColdTestContext)
    Repo.delete_all(ColdTestCase)
    Repo.delete_all(TestContext)
    Repo.delete_all(TestCase)

    if test_cases != [] do
      sequences =
        Enum.map(test_cases, fn tc -> (tc["context"] || []) ++ [tc["next_item"] || 0] end)

      {cases, context} = parse_test_data(sequences)
      insert_all_in_chunks(TestCase, cases)
      context = context |> Enum.uniq_by(fn r -> {r.case_id, r.pos} end)
      insert_all_in_chunks(TestContext, context)
    end

    if cold_test_cases != [] do
      sequences =
        Enum.map(cold_test_cases, fn tc -> (tc["context"] || []) ++ [tc["next_item"] || 0] end)

      {cases, context} = parse_test_data(sequences)
      insert_all_in_chunks(ColdTestCase, cases)
      context = context |> Enum.uniq_by(fn r -> {r.case_id, r.pos} end)
      insert_all_in_chunks(ColdTestContext, context)
    end

    :ok
  end

  defp parse_sequence_rows_from_list(sequences) do
    sequences
    |> Enum.with_index()
    |> Enum.flat_map(fn {elem, seq_id} ->
      {seq, timestamps} = extract_sequence_and_timestamps(elem)

      Enum.with_index(seq, fn item_id, pos ->
        row = %{seq_id: seq_id, pos: pos, item_id: item_id}

        time_ms =
          if timestamps && length(timestamps) > pos, do: Enum.at(timestamps, pos), else: nil

        if time_ms != nil, do: Map.put(row, :time_ms, time_ms), else: row
      end)
    end)
  end

  defp extract_sequence_and_timestamps(s) when is_list(s), do: {s, nil}

  defp extract_sequence_and_timestamps(%{"sequence" => s, "timestamps" => t})
       when is_list(s) and is_list(t),
       do: {s, t}

  defp extract_sequence_and_timestamps(%{"sequence" => s}) when is_list(s), do: {s, nil}
  defp extract_sequence_and_timestamps(_), do: {[], nil}

  @doc "Sync train_sequences.json and cold_train_sequences.json to DB. Paths can be nil to skip."
  def sync_sequences_from_json(train_path, cold_train_path) do
    Repo.delete_all(ColdTrainSequenceRow)
    Repo.delete_all(TrainSequenceRow)

    if train_path && File.regular?(train_path) do
      data = File.read!(train_path) |> Jason.decode!()

      rows =
        parse_sequence_rows(data["sequences"] || [], "train_sequence_rows")
        |> Enum.uniq_by(fn r -> {r.seq_id, r.pos} end)

      insert_all_in_chunks(TrainSequenceRow, rows)
    end

    if cold_train_path && File.regular?(cold_train_path) do
      data = File.read!(cold_train_path) |> Jason.decode!()

      rows =
        parse_sequence_rows(data["sequences"] || [], "cold_train_sequence_rows")
        |> Enum.uniq_by(fn r -> {r.seq_id, r.pos} end)

      insert_all_in_chunks(ColdTrainSequenceRow, rows)
    end

    :ok
  end

  @doc "Sync test_sequences.json and cold_test_sequences.json to DB (test_cases + test_context)."
  def sync_test_from_json(test_path, cold_test_path) do
    Repo.delete_all(ColdTestContext)
    Repo.delete_all(ColdTestCase)
    Repo.delete_all(TestContext)
    Repo.delete_all(TestCase)

    if test_path && File.regular?(test_path) do
      data = File.read!(test_path) |> Jason.decode!()
      test_raw = data["test_cases"] || data["sequences"] || []
      sequences = parse_test_sequences_from_cases(test_raw)
      {cases, context} = parse_test_data(sequences)
      insert_all_in_chunks(TestCase, cases)
      context = context |> Enum.uniq_by(fn r -> {r.case_id, r.pos} end)
      insert_all_in_chunks(TestContext, context)
    end

    if cold_test_path && File.regular?(cold_test_path) do
      data = File.read!(cold_test_path) |> Jason.decode!()
      test_raw = data["test_cases"] || data["sequences"] || []
      sequences = parse_test_sequences_from_cases(test_raw)
      {cases, context} = parse_test_data(sequences)
      insert_all_in_chunks(ColdTestCase, cases)
      context = context |> Enum.uniq_by(fn r -> {r.case_id, r.pos} end)
      insert_all_in_chunks(ColdTestContext, context)
    end

    :ok
  end

  defp insert_all_in_chunks(_schema, []), do: :ok

  defp insert_all_in_chunks(schema, rows) when is_list(rows) do
    rows
    |> Enum.chunk_every(@insert_chunk)
    |> Enum.each(&Repo.insert_all(schema, &1))
  end

  defp parse_test_sequences_from_cases(raw) do
    Enum.map(raw, fn
      %{"context" => ctx, "next_item" => next} when is_list(ctx) -> ctx ++ [next || 0]
      seq when is_list(seq) -> seq
      _ -> []
    end)
  end

  defp parse_sequence_rows(sequences, _table) do
    sequences
    |> Enum.with_index()
    |> Enum.flat_map(fn {elem, seq_id} ->
      {seq, timestamps} = extract_sequence_and_timestamps(elem)

      Enum.with_index(seq, fn item_id, pos ->
        row = %{seq_id: seq_id, pos: pos, item_id: item_id}

        time_ms =
          if timestamps && length(timestamps) > pos, do: Enum.at(timestamps, pos), else: nil

        if time_ms != nil, do: Map.put(row, :time_ms, time_ms), else: row
      end)
    end)
  end

  defp parse_test_data(sequences) do
    cases =
      sequences
      |> Enum.with_index()
      |> Enum.map(fn {seq, case_id} ->
        seq = if is_list(seq), do: seq, else: []
        next_item = List.last(seq)
        %{case_id: case_id, next_item: next_item || 0}
      end)

    context =
      sequences
      |> Enum.with_index()
      |> Enum.flat_map(fn {seq, case_id} ->
        seq = if is_list(seq), do: seq, else: []
        context_seq = Enum.drop(seq, -1)

        Enum.with_index(context_seq, fn item_id, pos ->
          %{case_id: case_id, pos: pos, item_id: item_id}
        end)
      end)

    {cases, context}
  end
end
