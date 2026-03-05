defmodule RecGPT.PretrainRunner do
  @moduledoc """
  Library entry point for running pretraining. Used by the Mix task and by StaffApi.

  Loads checkpoint, fixture, train sequences, and items; runs AxonTrain; writes updated
  params to the output directory. Call from `Mix.Tasks.Recgpt.Pretrain` or from
  `RecGPT.StaffApi.pretrain/1`.
  """
  alias RecGPT.AxonTrain
  alias RecGPT.Catalog.Scan
  alias RecGPT.CheckpointExport
  alias RecGPT.CheckpointLoader
  alias RecGPT.Embedding
  alias RecGPT.TestLoss

  @doc """
  Runs the pretrain pipeline.

  Options (keyword list):
    * `:ckpt_dir` - Checkpoint export dir (required)
    * `:fixture_path` - Path to fixture.json (required)
    * `:train_path` - Path to train_sequences.json (required)
    * `:items_path` - Path to items.json (required for non-empty sequences)
    * `:out_dir` - Output export dir (required)
    * `:limit` - Max items to use (default: fixture num_items)
    * `:iterations` - Max training steps (default: 100)
    * `:epochs` - Number of full passes (overrides iterations when set)
    * `:save_every` - Save checkpoint every N steps to <out_dir>/step_XXXX/ (0 = disable)
    * `:batch_size` - Batch size (default: 8)
    * `:learning_rate` - Learning rate (default: 1.0e-4)
    * `:log` - Log every N batches (default: 50; 0 to disable)
    * `:log_interval_sec` - Log progress at least every N seconds (default: 20)
    * `:eval_test_every` - Compute test loss every N steps (0 to disable)
    * `:test_path` - Path to test_sequences.json (required when eval_test_every > 0)
    * `:mtp_loss_weight` - Weight for MTP loss over last 4 positions (default 1.0). Set 0 to use only shifted CE.
    * `:resource_check_opts` - Options for ResourceCheck (e.g. max_memory_mb)

  Returns `:ok` on success, or `{:error, reason}`.
  """
  @spec run(keyword()) :: :ok | {:error, term()}
  def run(opts) do
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:bumblebee)

    ckpt_dir = Keyword.fetch!(opts, :ckpt_dir)
    fixture_path = Keyword.fetch!(opts, :fixture_path)
    train_path = Keyword.fetch!(opts, :train_path)
    items_path = Keyword.get(opts, :items_path)
    out_dir = Keyword.fetch!(opts, :out_dir)

    batch_size = Keyword.get(opts, :batch_size, 8)
    epochs = Keyword.get(opts, :epochs)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-4)
    log_every = Keyword.get(opts, :log, 50)
    log_interval_sec = Keyword.get(opts, :log_interval_sec, 20)
    resource_check_opts = Keyword.get(opts, :resource_check_opts, [])

    with :ok <- ensure_regular_file!(fixture_path, "fixture"),
         :ok <- ensure_train_source!(train_path),
         :ok <- ensure_dir!(ckpt_dir, "ckpt_dir"),
         {:ok, params} <- load_checkpoint(ckpt_dir),
         {:ok, fixture} <- load_fixture(fixture_path),
         {:ok, train_data} <- load_train_data(train_path) do
      empty? = train_data_empty?(train_data)
      if empty? do
        File.mkdir_p!(out_dir)
        CheckpointExport.write_export(params, out_dir)
        :ok
      else
        with :ok <- require_items_source!(items_path),
             {:ok, token_id_list, fixture_num_items} <- fixture_token_list(fixture),
             {:ok, token_id_list} <- maybe_token_list_from_db(token_id_list, fixture_num_items),
             {:ok, item_embeddings, _n} <-
               load_item_embeddings(items_path, fixture_num_items, opts) do
          epochs = epochs || 1
          steps_per_epoch = steps_per_epoch_from_train_data(train_data, batch_size)
          iterations =
            if Keyword.get(opts, :epochs),
              do: epochs * steps_per_epoch,
              else: Keyword.get(opts, :iterations, 100)

          stream =
            build_train_stream(train_data, token_id_list, item_embeddings,
              batch_size: batch_size,
              epochs: epochs,
              shuffle: true
            )

          save_every = opts[:save_every] |> Kernel.||(0)

          save_fn =
            if save_every > 0 do
              fn step, params ->
                step_dir =
                  Path.join(
                    out_dir,
                    "step_#{String.pad_leading(Integer.to_string(step), 6, "0")}"
                  )

                File.mkdir_p!(step_dir)
                CheckpointExport.write_export(params, step_dir)
                require Logger
                Logger.info("Saved checkpoint at step #{step} to #{step_dir}")
              end
            else
              nil
            end

          eval_test_every = opts[:eval_test_every] |> Kernel.||(0)
          test_path = opts[:test_path]

          eval_test_fn =
            if eval_test_every > 0 and test_path && File.regular?(test_path) do
              fn params ->
                TestLoss.compute(params, token_id_list, item_embeddings, test_path,
                  batch_size: batch_size,
                  limit: opts[:limit]
                )
              end
            else
              nil
            end

          mtp_loss_weight = Keyword.get(opts, :mtp_loss_weight, 1.0)

          train_opts = [
            iterations: iterations,
            log: log_every,
            log_interval_sec: log_interval_sec,
            learning_rate: learning_rate,
            resource_check_interval: 5,
            resource_check_opts: resource_check_opts,
            mtp_loss_weight: mtp_loss_weight
          ]

          train_opts =
            if save_every > 0,
              do: Keyword.merge(train_opts, save_every: save_every, save_fn: save_fn),
              else: train_opts

          train_opts =
            if eval_test_every > 0 and eval_test_fn,
              do:
                Keyword.merge(train_opts,
                  eval_test_every: eval_test_every,
                  eval_test_fn: eval_test_fn
                ),
              else: train_opts

          trained = AxonTrain.run(stream, params, train_opts)

          File.mkdir_p!(out_dir)
          CheckpointExport.write_export(trained, out_dir)
          :ok
        end
      end
    end
  end

  defp require_items_source!(nil), do: {:error, :items_source_required_for_pretrain}
  defp require_items_source!(:db), do: :ok
  defp require_items_source!("db"), do: :ok
  defp require_items_source!(path) when is_binary(path) do
    if File.regular?(path), do: :ok, else: {:error, {:missing_file, path}}
  end

  defp ensure_train_source!(:db), do: :ok
  defp ensure_train_source!("db"), do: :ok

  defp ensure_train_source!(path) when is_binary(path),
    do: ensure_regular_file!(path, "train_sequences")

  defp ensure_regular_file!(path, _name) when is_binary(path) do
    if File.regular?(path), do: :ok, else: {:error, {:missing_file, path}}
  end

  defp ensure_regular_file!(nil, name), do: {:error, {:missing_file, name, nil}}

  defp ensure_dir!(path, name) do
    if File.dir?(path), do: :ok, else: {:error, {:missing_dir, name, path}}
  end

  defp load_checkpoint(dir) do
    manifest = Path.join(dir, "manifest.json")

    if File.regular?(manifest),
      do: {:ok, CheckpointLoader.load_from_export(dir)},
      else: {:error, {:checkpoint_not_found, dir}}
  end

  defp load_fixture(path) do
    raw = File.read!(path) |> Jason.decode!()
    {:ok, raw}
  rescue
    e -> {:error, e}
  end

  defp load_train_data(:db), do: {:ok, {:db, Scan.load_train_seq_ids()}}
  defp load_train_data("db"), do: {:ok, {:db, Scan.load_train_seq_ids()}}

  defp load_train_data(path) when is_binary(path) do
    case load_train_sequences(path) do
      {:ok, sequences, timestamps} -> {:ok, {:file, sequences, timestamps}}
      err -> err
    end
  end

  defp train_data_empty?({:db, seq_ids}), do: seq_ids == []
  defp train_data_empty?({:file, sequences, _}), do: sequences == []

  defp steps_per_epoch_from_train_data({:db, seq_ids}, batch_size),
    do: div(length(seq_ids) + batch_size - 1, batch_size)
  defp steps_per_epoch_from_train_data({:file, sequences, _}, batch_size),
    do: div(length(sequences) + batch_size - 1, batch_size)

  defp build_train_stream({:db, seq_ids}, token_id_list, item_embeddings, opts) do
    AxonTrain.stream_batches_from_db(seq_ids, token_id_list, item_embeddings, opts)
  end

  defp build_train_stream({:file, sequences, timestamps}, token_id_list, item_embeddings, opts) do
    AxonTrain.stream_batches(sequences, token_id_list, item_embeddings,
      Keyword.put(opts, :timestamps, timestamps)
    )
  end

  defp maybe_token_list_from_db(token_id_list, _num_items) when is_list(token_id_list) and token_id_list != [] do
    {:ok, token_id_list}
  end

  defp maybe_token_list_from_db(_token_id_list, num_items) when num_items > 0 do
    {:ok, Scan.load_token_id_list_from_db(num_items)}
  end

  defp maybe_token_list_from_db(token_id_list, _), do: {:ok, token_id_list || []}

  defp load_train_sequences(:db), do: load_train_sequences_from_db()
  defp load_train_sequences("db"), do: load_train_sequences_from_db()

  defp load_train_sequences(path) when is_binary(path) do
    raw = File.read!(path) |> Jason.decode!()
    raw_seqs = raw["sequences"] || []
    sequences = Enum.map(raw_seqs, &normalize_sequence/1)
    timestamps =
      Enum.map(raw_seqs, fn
        %{"timestamps" => t} when is_list(t) -> t
        _ -> nil
      end)
    timestamps = if Enum.any?(timestamps, & &1), do: timestamps, else: nil
    {:ok, sequences, timestamps}
  rescue
    e -> {:error, e}
  end

  defp normalize_sequence(s) when is_list(s), do: s
  defp normalize_sequence(%{"sequence" => s}) when is_list(s), do: s
  defp normalize_sequence(_), do: []

  defp load_train_sequences_from_db do
    import Ecto.Query
    alias RecGPT.Catalog.TrainSequenceRow
    alias RecGPT.Repo

    rows =
      from(r in TrainSequenceRow, order_by: [asc: r.seq_id, asc: r.pos])
      |> Repo.all()

    {sequences, timestamps} =
      rows
      |> Enum.chunk_by(& &1.seq_id)
      |> Enum.map(fn chunk ->
        seq = Enum.map(chunk, & &1.item_id)
        ts = Enum.map(chunk, & &1.time_ms)
        has_ts = Enum.any?(ts, & &1)
        {seq, if(has_ts, do: ts, else: nil)}
      end)
      |> Enum.unzip()

    timestamps = if Enum.any?(timestamps, & &1), do: timestamps, else: nil
    {:ok, sequences, timestamps}
  end

  defp fixture_token_list(fixture) do
    token_id_list =
      (fixture["token_id_list"] || [])
      |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

    num_items = fixture["num_items"] || length(token_id_list)
    {:ok, token_id_list, num_items}
  end

  defp load_item_embeddings(:db, fixture_num_items, opts), do: load_item_embeddings_from_db(fixture_num_items, opts)
  defp load_item_embeddings("db", fixture_num_items, opts), do: load_item_embeddings_from_db(fixture_num_items, opts)

  defp load_item_embeddings(items_path, fixture_num_items, opts) when is_binary(items_path) do
    raw = File.read!(items_path) |> Jason.decode!()
    items = raw["items"] || []
    items_n = raw["num_items"] || length(items)
    limit = Keyword.get(opts, :limit)

    n =
      if limit,
        do: min(items_n, fixture_num_items) |> min(limit),
        else: min(items_n, fixture_num_items)

    item_text_dict =
      items
      |> Enum.take(n)
      |> Enum.with_index()
      |> Map.new(fn {item, idx} -> {idx, Embedding.recgpt_item_text(item)} end)

    item_embeddings = Embedding.encode_item_text_dict(item_text_dict)
    {:ok, item_embeddings, n}
  rescue
    e -> {:error, e}
  end

  defp load_item_embeddings_from_db(fixture_num_items, opts) do
    import Ecto.Query
    alias RecGPT.Catalog.Item
    alias RecGPT.Catalog.ItemEmbeddingText
    alias RecGPT.Repo

    total = Repo.aggregate(Item, :count, :item_id)
    n = min(total, fixture_num_items)
    n = if limit = Keyword.get(opts, :limit), do: min(n, limit), else: n

    items =
      from(i in Item, order_by: [asc: i.item_id], limit: ^n)
      |> Repo.all()

    embed_map =
      from(e in ItemEmbeddingText, where: e.item_id in ^Enum.map(items, & &1.item_id))
      |> Repo.all()
      |> Map.new(&{&1.item_id, &1.embedding_text})

    item_text_dict =
      items
      |> Enum.with_index()
      |> Map.new(fn {item, idx} ->
        text = Map.get(embed_map, item.item_id) || item.title
        {idx, Embedding.recgpt_item_text(%{title: text})}
      end)

    item_embeddings = Embedding.encode_item_text_dict(item_text_dict)
    {:ok, item_embeddings, map_size(item_text_dict)}
  end
end
