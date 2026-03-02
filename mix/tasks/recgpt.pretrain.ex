defmodule Mix.Tasks.Recgpt.Pretrain do
  @shortdoc "Pretrain on train_sequences with fixture and checkpoint; write updated export"
  @moduledoc """
  Loads checkpoint, train_sequences.json, fixture (token_id_list), and item embeddings;
  runs AxonTrain.stream_batches + AxonTrain.run; always writes updated params to --out.

  Pipeline: Fetch → build_fixture → pretrain → eval (with --test and --cold-test).

  ## Options
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--fixture` - Fixture JSON path (default: data/steam/fixture.json)
    * `--train` - train_sequences.json path (default: data/steam/train_sequences.json)
    * `--items` - items.json for building embeddings (default: data/steam/items.json)
    * `--limit` - Max items to encode for training (default: fixture num_items). Prevents loading 30+ GB when items.json is large.
    * `--out` - Output export dir (required)
    * `--iterations` - Max training steps (default: 100)
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50; 0 to disable)
    * `--log-interval-sec` - Log progress at least every N seconds (default: 20; 0 to disable)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          ckpt: :string,
          fixture: :string,
          train: :string,
          items: :string,
          limit: :integer,
          out: :string,
          iterations: :integer,
          batch_size: :integer,
          learning_rate: :float,
          log: :integer,
          log_interval_sec: :integer
        ]
      )

    ckpt_dir = opts[:ckpt] || resolve("data/recgpt_ckpt_export")
    fixture_path = opts[:fixture] || resolve("data/steam/fixture.json")
    train_path = opts[:train] || resolve("data/steam/train_sequences.json")
    items_path = opts[:items] || resolve("data/steam/items.json")

    out_dir =
      case opts[:out] do
        nil -> Mix.raise("--out DIR is required")
        "" -> Mix.raise("--out DIR is required")
        s -> resolve(s)
      end

    iterations = opts[:iterations] || 100
    batch_size = opts[:batch_size] || 8
    learning_rate = opts[:learning_rate] || 1.0e-4
    log_every = opts[:log] || 50
    log_interval_sec = opts[:log_interval_sec]
    log_interval_sec = if is_integer(log_interval_sec), do: log_interval_sec, else: 20

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("checkpoint not found: #{ckpt_dir}")
    end

    unless File.regular?(fixture_path) do
      Mix.raise("fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.regular?(train_path) do
      Mix.raise("train sequences not found: #{train_path}. Run mix recgpt.fetch_steam first.")
    end

    runner_opts = [
      ckpt_dir: ckpt_dir,
      fixture_path: fixture_path,
      train_path: train_path,
      items_path: items_path,
      out_dir: out_dir,
      limit: opts[:limit],
      iterations: iterations,
      batch_size: batch_size,
      learning_rate: learning_rate,
      log: log_every,
      log_interval_sec: log_interval_sec,
      resource_check_opts: pretrain_resource_check_opts()
    ]

    case RecGPT.PretrainRunner.run(runner_opts) do
      :ok ->
        Mix.shell().info("Done.")
        :ok
      {:error, reason} ->
        Mix.raise("Pretrain failed: #{inspect(reason)}")
    end
  end

  defp pretrain_resource_check_opts do
    case System.get_env("RECGPT_MAX_MEMORY_MB") do
      nil ->
        []

      s ->
        case Integer.parse(s) do
          {n, _} when n > 0 -> [max_memory_mb: n]
          _ -> []
        end
    end
  end

  defp resolve(path) when is_binary(path) do
    if absolute_path?(path), do: path, else: Path.expand(path, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
