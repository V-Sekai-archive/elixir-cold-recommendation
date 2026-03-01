defmodule RecGPT.FixtureBuild do
  @moduledoc """
  Build fixture.json from items.json: Embedding.encode_item_text_dict + FSQ → token_id_list.

  Used by `mix recgpt.build_fixture`. Output format matches Serve.load_fixture (num_items, token_id_list).
  When RECGPT_SQLITE_PATH is set (or opts[:sqlite]), flushes items, embeddings, and tokens to SQLite per batch.
  """

  alias RecGPT.Catalog.Sync
  alias RecGPT.CheckpointLoader
  alias RecGPT.Embedding
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder

  @sqlite_batch_size 100

  @doc """
  Builds fixture from items path and checkpoint.

  - Reads items.json → item_text_dict (id => title for 0..num_items-1).
  - Encodes via Embedding.encode_item_text_dict/1 → {num_items, 768}.
  - Loads FSQ params from ckpt_dir.
  - Encodes embeddings to token_id_list via FSQEncoder.encode_embeddings_to_token_id_list/3.
  - When sqlite: processes in batches and flushes items, item_embeddings, item_tokens to SQLite each batch.
  - Returns %{"num_items" => n, "token_id_list" => token_id_list}.
  """
  @spec build(String.t(), String.t(), keyword()) :: %{
          String.t() => non_neg_integer() | [[non_neg_integer()]]
        }
  def build(items_path, ckpt_dir, opts \\ []) do
    item_text_dict = load_item_text_dict(items_path, opts[:limit])
    sqlite? = opts[:sqlite] || System.get_env("RECGPT_SQLITE_PATH") != nil

    if sqlite? do
      build_with_sqlite(item_text_dict, ckpt_dir, items_path)
    else
      build_in_memory(item_text_dict, ckpt_dir)
    end
  end

  defp build_in_memory(item_text_dict, ckpt_dir) do
    embeddings = Embedding.encode_item_text_dict(item_text_dict)
    {num_items, _} = Nx.shape(embeddings)
    num_items = if is_tuple(num_items), do: elem(num_items, 0), else: num_items
    fsq_params = load_fsq_params(ckpt_dir)
    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)
    %{"num_items" => num_items, "token_id_list" => token_id_list}
  end

  defp build_with_sqlite(item_text_dict, ckpt_dir, items_path) do
    Application.ensure_all_started(:recgpt)
    fsq_params = load_fsq_params(ckpt_dir)
    ids = item_text_dict |> Map.keys() |> Enum.sort()
    num_items = length(ids)

    {token_id_list, _cleared} =
      ids
      |> Enum.chunk_every(@sqlite_batch_size)
      |> Enum.reduce({[], false}, fn batch_ids, {acc, cleared} ->
        batch_dict = Map.take(item_text_dict, batch_ids)
        embeddings = Embedding.encode_item_text_dict(batch_dict)
        batch_tokens = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)

        unless cleared, do: Sync.clear_catalog_tables()

        entries_items =
          Enum.map(batch_ids, fn id -> %{item_id: id, title: Map.fetch!(item_text_dict, id)} end)

        entries_embeddings =
          batch_ids
          |> Enum.with_index()
          |> Enum.map(fn {id, i} ->
            bin = Nx.slice(embeddings, [i, 0], [1, 768]) |> Nx.to_binary()
            %{item_id: id, embedding: bin}
          end)

        entries_tokens =
          Enum.zip(batch_ids, batch_tokens)
          |> Enum.map(fn {id, [t0, t1, t2, t3]} ->
            %{item_id: id, t0: t0, t1: t1, t2: t2, t3: t3}
          end)

        Sync.insert_items(entries_items)
        Sync.insert_item_embeddings(entries_embeddings)
        Sync.insert_item_tokens(entries_tokens)

        {acc ++ batch_tokens, true}
      end)

    base = Path.dirname(items_path)

    Sync.sync_sequences_from_json(
      Path.join(base, "train_sequences.json"),
      Path.join(base, "cold_train_sequences.json")
    )

    Sync.sync_test_from_json(
      Path.join(base, "test_sequences.json"),
      Path.join(base, "cold_test_sequences.json")
    )

    %{"num_items" => num_items, "token_id_list" => token_id_list}
  end

  @doc "Writes fixture map to path (JSON)."
  @spec write_fixture(map(), String.t()) :: :ok
  def write_fixture(fixture, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(fixture, pretty: true))
    :ok
  end

  defp load_item_text_dict(path, limit) do
    raw = File.read!(path) |> Jason.decode!()
    items = raw["items"] || []
    num_items = raw["num_items"] || length(items)
    num_items = if limit, do: min(num_items, limit), else: num_items

    items
    |> Enum.take(num_items)
    |> Enum.with_index()
    |> Map.new(fn {item, idx} ->
      title = item["title"] || item["text"] || item["raw"] || ""
      {idx, title}
    end)
  end

  defp load_fsq_params(ckpt_dir) do
    ckpt_params = CheckpointLoader.load_from_export(ckpt_dir)
    params = FSQ.load_params(ckpt_params)

    if fsq_params_ok?(params) do
      params
    else
      raise "FSQ params not found in checkpoint #{ckpt_dir}. " <>
              "Use a checkpoint that includes FSQ (e.g. project_in/kernel or fsq.project_in.weight)."
    end
  end

  defp fsq_params_ok?(%{"project_in" => %{"kernel" => k}, "project_out" => %{"kernel" => o}})
       when not is_nil(k) and not is_nil(o),
       do: true

  defp fsq_params_ok?(_), do: false
end
