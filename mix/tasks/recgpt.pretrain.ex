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
    * `--out` - Output export dir (required)
    * `--iterations` - Max training steps (default: 100)
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50; 0 to disable)
    * `--timeout` - Max seconds to run (default: 86400)
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
          out: :string,
          iterations: :integer,
          batch_size: :integer,
          learning_rate: :float,
          log: :integer,
          timeout: :integer
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
    timeout_ms = (opts[:timeout] || 86_400) * 1000

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("checkpoint not found: #{ckpt_dir}")
    end

    unless File.regular?(fixture_path) do
      Mix.raise("fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.regular?(train_path) do
      Mix.raise("train sequences not found: #{train_path}. Run mix recgpt.fetch_steam first.")
    end

    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:bumblebee)

    Mix.shell().info("Loading checkpoint from #{ckpt_dir}...")
    params = RecGPT.CheckpointLoader.load_from_export(ckpt_dir)

    fixture = File.read!(fixture_path) |> Jason.decode!()

    token_id_list =
      (fixture["token_id_list"] || []) |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

    _num_items = fixture["num_items"] || length(token_id_list)

    train_raw = File.read!(train_path) |> Jason.decode!()
    sequences = train_raw["sequences"] || []

    if sequences == [] do
      Mix.shell().info("No train sequences; writing checkpoint unchanged to #{out_dir}")
      RecGPT.CheckpointExport.write_export(params, out_dir)
      Mix.shell().info("Done.")
    else
      unless File.regular?(items_path) do
        Mix.raise("items not found: #{items_path}")
      end

      Mix.shell().info(
        "Pretrain (timeout: #{div(timeout_ms, 1000)}s, up to #{iterations} steps)..."
      )

      task =
        Task.async(fn ->
          raw = File.read!(items_path) |> Jason.decode!()
          items = raw["items"] || []
          n = raw["num_items"] || length(items)

          item_text_dict =
            items
            |> Enum.take(n)
            |> Enum.with_index()
            |> Map.new(fn {item, idx} -> {idx, item["title"] || item["text"] || ""} end)

          item_embeddings = RecGPT.Embedding.encode_item_text_dict(item_text_dict)

          stream =
            RecGPT.AxonTrain.stream_batches(sequences, token_id_list, item_embeddings,
              batch_size: batch_size,
              epochs: 1,
              shuffle: true
            )

          trained =
            RecGPT.AxonTrain.run(stream, params,
              iterations: iterations,
              log: log_every,
              learning_rate: learning_rate
            )

          RecGPT.CheckpointExport.write_export(trained, out_dir)
          :ok
        end)

      try do
        Task.await(task, timeout_ms)
        Mix.shell().info("Done.")
      rescue
        e in [Task.TimeoutError] ->
          Mix.raise("Pretrain timed out after #{div(timeout_ms, 1000)}s")
      end
    end

    :ok
  end

  defp resolve(path) when is_binary(path) do
    if absolute_path?(path), do: path, else: Path.expand(path, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
