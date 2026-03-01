defmodule RecGPT.Catalog.Sync do
  @moduledoc """
  Flush catalog and sequence data to SQLite (ETNF tables).
  Used by FixtureBuild when RECGPT_SQLITE_PATH is set; flush as soon as each batch is ready.
  """

  alias RecGPT.Catalog.ColdTestCase
  alias RecGPT.Catalog.ColdTestContext
  alias RecGPT.Catalog.ColdTrainSequenceRow
  alias RecGPT.Catalog.{Item, ItemEmbedding, ItemToken}
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

  @doc "Insert item_tokens batch: list of %{item_id: id, t0: n, t1: n, t2: n, t3: n}."
  def insert_item_tokens(entries) when is_list(entries) and entries != [] do
    Repo.insert_all(ItemToken, entries)
  end

  @insert_chunk 1000

  @doc "Sync train_sequences.json and cold_train_sequences.json to DB. Paths can be nil to skip."
  def sync_sequences_from_json(train_path, cold_train_path) do
    Repo.delete_all(ColdTrainSequenceRow)
    Repo.delete_all(TrainSequenceRow)

    if train_path && File.regular?(train_path) do
      data = File.read!(train_path) |> Jason.decode!()
      rows = parse_sequence_rows(data["sequences"] || [], "train_sequence_rows")
      insert_all_in_chunks(TrainSequenceRow, rows)
    end

    if cold_train_path && File.regular?(cold_train_path) do
      data = File.read!(cold_train_path) |> Jason.decode!()
      rows = parse_sequence_rows(data["sequences"] || [], "cold_train_sequence_rows")
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
      {cases, context} = parse_test_data(data["test_cases"] || data["sequences"] || [])
      insert_all_in_chunks(TestCase, cases)
      insert_all_in_chunks(TestContext, context)
    end

    if cold_test_path && File.regular?(cold_test_path) do
      data = File.read!(cold_test_path) |> Jason.decode!()
      {cases, context} = parse_test_data(data["test_cases"] || data["sequences"] || [])
      insert_all_in_chunks(ColdTestCase, cases)
      insert_all_in_chunks(ColdTestContext, context)
    end

    :ok
  end

  defp insert_all_in_chunks(_schema, []), do: :ok

  defp insert_all_in_chunks(schema, rows) do
    rows
    |> Enum.chunk_every(@insert_chunk)
    |> Enum.each(&Repo.insert_all(schema, &1))
  end

  defp parse_sequence_rows(sequences, _table) do
    sequences
    |> Enum.with_index()
    |> Enum.flat_map(fn {seq, seq_id} ->
      seq = if is_list(seq), do: seq, else: []

      Enum.with_index(seq, fn item_id, pos ->
        %{seq_id: seq_id, pos: pos, item_id: item_id}
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
