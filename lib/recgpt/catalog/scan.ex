defmodule RecGPT.Catalog.Scan do
  @moduledoc """
  Stream the entire catalogue in constant memory.

  Use when you need to iterate over all items or all item tokens without loading
  the full catalogue into RAM (e.g. build fixture, export, stats, batch jobs).

  At serve time, `RecGPT.Serve.load_fixture_from_db/2` also streams item tokens
  from the DB to build the trie, so the entire catalogue is scanned with bounded memory.

  ## Examples

      # Stream all items (id + title)
      RecGPT.Catalog.Scan.stream_items()
      |> Enum.each(fn item -> ... end)

      # Stream all item tokens (id + [t0,t1,t2,t3])
      RecGPT.Catalog.Scan.stream_item_tokens()
      |> Enum.reduce(acc, fn {id, tokens}, acc -> ... end)

      # Process in batches of 500 (constant memory)
      RecGPT.Catalog.Scan.stream_item_tokens_chunked(batch_size: 500)
      |> Enum.each(fn batch -> ... end)
  """

  import Ecto.Query
  alias RecGPT.Catalog.Item
  alias RecGPT.Catalog.ItemToken
  alias RecGPT.Repo

  @default_batch_size 500

  @doc """
  Stream all items from the catalogue, ordered by item_id.
  Constant memory: one row at a time. Requires RECGPT_SQLITE_PATH and migrated DB.
  """
  @spec stream_items(keyword()) :: Enumerable.t()
  def stream_items(opts \\ []) do
    repo = Keyword.get(opts, :repo, Repo)

    from(i in Item, order_by: [asc: i.item_id], select: %{item_id: i.item_id, title: i.title})
    |> repo.stream()
  end

  @doc """
  Stream all item tokens from the catalogue, ordered by item_id.
  Each element is `{item_id, [t0, t1, t2, t3]}`. Constant memory.
  """
  @spec stream_item_tokens(keyword()) :: Enumerable.t()
  def stream_item_tokens(opts \\ []) do
    repo = Keyword.get(opts, :repo, Repo)

    from(t in ItemToken, order_by: [asc: t.item_id], select: {t.item_id, t.t0, t.t1, t.t2, t.t3})
    |> repo.stream()
    |> Stream.map(fn {id, t0, t1, t2, t3} ->
      {id, [t0 || 0, t1 || 0, t2 || 0, t3 || 0]}
    end)
  end

  @doc """
  Stream item tokens in batches. Yields lists of `{item_id, [t0,t1,t2,t3]}`.
  Use to scan the entire catalogue with bounded memory and batch processing.
  """
  @spec stream_item_tokens_chunked(keyword()) :: Enumerable.t()
  def stream_item_tokens_chunked(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    stream_item_tokens(opts) |> Stream.chunk_every(batch_size)
  end

  @doc """
  Stream items in batches. Yields lists of `%{item_id: id, title: title}`.
  """
  @spec stream_items_chunked(keyword()) :: Enumerable.t()
  def stream_items_chunked(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    stream_items(opts) |> Stream.chunk_every(batch_size)
  end

  @doc """
  Count items in the catalogue without loading them (constant memory).
  """
  @spec count_items(keyword()) :: non_neg_integer()
  def count_items(opts \\ []) do
    repo = Keyword.get(opts, :repo, Repo)
    repo.aggregate(Item, :count, :item_id)
  end

  # ---------------------------------------------------------------------------
  # Train sequences (constant memory: load only seq_ids, then fetch rows per batch)
  # ---------------------------------------------------------------------------

  @doc """
  Load all train sequence IDs from the DB. Small memory: one integer per sequence.
  Use with `load_sequences_by_seq_ids/2` to stream batches for training (e.g. 5 epochs).
  """
  @spec load_train_seq_ids(keyword()) :: [non_neg_integer()]
  def load_train_seq_ids(opts \\ []) do
    repo = Keyword.get(opts, :repo, Repo)
    alias RecGPT.Catalog.TrainSequenceRow

    from(r in TrainSequenceRow, distinct: true, select: r.seq_id, order_by: [asc: r.seq_id])
    |> repo.all()
  end

  @doc """
  Load sequences and timestamps for the given seq_ids. One query; order matches seq_ids.
  Returns `{sequences, timestamps}` where sequences is a list of item_id lists and
  timestamps is a list of time_ms lists (or nil per sequence). Use for one batch.
  """
  @spec load_sequences_by_seq_ids([non_neg_integer()], keyword()) ::
          {[[non_neg_integer()]], [nil | [non_neg_integer()]]}
  def load_sequences_by_seq_ids(seq_ids, opts \\ []) when is_list(seq_ids) do
    if seq_ids == [] do
      {[], []}
    else
      repo = Keyword.get(opts, :repo, Repo)
      alias RecGPT.Catalog.TrainSequenceRow

      rows =
        from(r in TrainSequenceRow,
          where: r.seq_id in ^seq_ids,
          order_by: [asc: r.seq_id, asc: r.pos],
          select: {r.seq_id, r.pos, r.item_id, r.time_ms}
        )
        |> repo.all()

      # Group by seq_id; rows are already ordered by pos from query
      by_seq =
        Enum.group_by(rows, fn {sid, _pos, _iid, _t} -> sid end, fn {_sid, _pos, iid, t} ->
          {iid, t}
        end)

      sequences =
        Enum.map(seq_ids, fn sid ->
          Map.get(by_seq, sid, []) |> Enum.map(&elem(&1, 0))
        end)

      timestamps =
        Enum.map(seq_ids, fn sid ->
          ts = Map.get(by_seq, sid, []) |> Enum.map(&elem(&1, 1))
          if Enum.any?(ts, & &1), do: ts, else: nil
        end)

      {sequences, timestamps}
    end
  end

  @doc """
  Build token_id_list (list of [t0,t1,t2,t3] per item_id) from DB for training.
  Use when fixture was built with SQLite and has token_id_list: [].
  Streams from ItemToken so memory is bounded by the stream; result is O(num_items).
  """
  @spec load_token_id_list_from_db(non_neg_integer(), keyword()) :: [[non_neg_integer()]]
  def load_token_id_list_from_db(num_items, opts \\ []) when num_items >= 0 do
    if num_items == 0 do
      []
    else
      stream_item_tokens(opts)
      |> Enum.take(num_items)
      |> Enum.map(fn {_id, tokens} -> tokens end)
    end
  end
end
