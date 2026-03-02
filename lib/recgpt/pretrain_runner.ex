defmodule RecGPT.PretrainRunner do
  @moduledoc """
  Library entry point for running pretraining. Used by the Mix task and by StaffApi.

  Loads checkpoint, fixture, train sequences, and items; runs AxonTrain; writes updated
  params to the output directory. Call from `Mix.Tasks.Recgpt.Pretrain` or from
  `RecGPT.StaffApi.pretrain/1`.
  """
  alias RecGPT.AxonTrain
  alias RecGPT.CheckpointExport
  alias RecGPT.CheckpointLoader
  alias RecGPT.Embedding

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
    * `:batch_size` - Batch size (default: 8)
    * `:learning_rate` - Learning rate (default: 1.0e-4)
    * `:log` - Log every N batches (default: 50; 0 to disable)
    * `:log_interval_sec` - Log progress at least every N seconds (default: 20)
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

    iterations = Keyword.get(opts, :iterations, 100)
    batch_size = Keyword.get(opts, :batch_size, 8)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-4)
    log_every = Keyword.get(opts, :log, 50)
    log_interval_sec = Keyword.get(opts, :log_interval_sec, 20)
    resource_check_opts = Keyword.get(opts, :resource_check_opts, [])

    with :ok <- ensure_regular_file!(fixture_path, "fixture"),
         :ok <- ensure_regular_file!(train_path, "train_sequences"),
         :ok <- ensure_dir!(ckpt_dir, "ckpt_dir"),
         {:ok, params} <- load_checkpoint(ckpt_dir),
         {:ok, fixture} <- load_fixture(fixture_path),
         {:ok, sequences} <- load_train_sequences(train_path) do
      if sequences == [] do
        File.mkdir_p!(out_dir)
        CheckpointExport.write_export(params, out_dir)
        :ok
      else
        with {:ok, _} <- require_items_path(items_path),
             :ok <- ensure_regular_file!(items_path, "items"),
             {:ok, token_id_list, fixture_num_items} <- fixture_token_list(fixture),
             {:ok, item_embeddings, _n} <- load_item_embeddings(items_path, fixture_num_items, opts) do
          stream =
            AxonTrain.stream_batches(sequences, token_id_list, item_embeddings,
              batch_size: batch_size,
              epochs: 1,
              shuffle: true
            )

          trained =
            AxonTrain.run(stream, params,
              iterations: iterations,
              log: log_every,
              log_interval_sec: log_interval_sec,
              learning_rate: learning_rate,
              resource_check_interval: 5,
              resource_check_opts: resource_check_opts
            )

          File.mkdir_p!(out_dir)
          CheckpointExport.write_export(trained, out_dir)
          :ok
        end
      end
    end
  end

  defp require_items_path(nil), do: {:error, :items_path_required_for_pretrain}
  defp require_items_path(path) when is_binary(path), do: {:ok, path}

  defp ensure_regular_file!(path, _name) when is_binary(path) do
    if File.regular?(path), do: :ok, else: {:error, {:missing_file, path}}
  end

  defp ensure_regular_file!(nil, name), do: {:error, {:missing_file, name, nil}}

  defp ensure_dir!(path, name) do
    if File.dir?(path), do: :ok, else: {:error, {:missing_dir, name, path}}
  end

  defp load_checkpoint(dir) do
    manifest = Path.join(dir, "manifest.json")
    if File.regular?(manifest), do: {:ok, CheckpointLoader.load_from_export(dir)}, else: {:error, {:checkpoint_not_found, dir}}
  end

  defp load_fixture(path) do
    raw = File.read!(path) |> Jason.decode!()
    {:ok, raw}
  rescue
    e -> {:error, e}
  end

  defp load_train_sequences(path) do
    raw = File.read!(path) |> Jason.decode!()
    sequences = raw["sequences"] || []
    {:ok, sequences}
  rescue
    e -> {:error, e}
  end

  defp fixture_token_list(fixture) do
    token_id_list =
      (fixture["token_id_list"] || [])
      |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

    num_items = fixture["num_items"] || length(token_id_list)
    {:ok, token_id_list, num_items}
  end

  defp load_item_embeddings(items_path, fixture_num_items, opts) do
    raw = File.read!(items_path) |> Jason.decode!()
    items = raw["items"] || []
    items_n = raw["num_items"] || length(items)
    limit = Keyword.get(opts, :limit)
    n = if limit, do: min(items_n, fixture_num_items) |> min(limit), else: min(items_n, fixture_num_items)

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
end
