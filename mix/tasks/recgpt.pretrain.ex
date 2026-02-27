defmodule Mix.Tasks.Recgpt.Pretrain do
  @shortdoc "Pretrain on train_sequences with fixture and checkpoint; write updated export"
  @moduledoc """
  Loads checkpoint, train_sequences.json, fixture (token_id_list), and item embeddings;
  runs AxonTrain.stream_batches + AxonTrain.run; always writes updated params to --out.

  Pipeline: Fetch → build_fixture → pretrain → eval (with --test and --cold-test).

  ## Options
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--fixture` - Fixture JSON path (default: data/clickstream/fixture.json)
    * `--train` - train_sequences.json path (default: data/clickstream/train_sequences.json)
    * `--items` - items.json for building embeddings (default: data/clickstream/items.json); ignored if --embeddings given
    * `--embeddings` - Optional path to precomputed embeddings (Nx.serialize); overrides --items
    * `--out` - Output export dir (required)
    * `--iterations` - Max training steps (default: 100)
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50; 0 to disable)
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
          embeddings: :string,
          out: :string,
          iterations: :integer,
          batch_size: :integer,
          learning_rate: :float,
          log: :integer
        ]
      )

    ckpt_dir = opts[:ckpt] || resolve("data/recgpt_ckpt_export")
    fixture_path = opts[:fixture] || resolve("data/clickstream/fixture.json")
    train_path = opts[:train] || resolve("data/clickstream/train_sequences.json")
    items_path = opts[:items] || resolve("data/clickstream/items.json")
    embeddings_path = opts[:embeddings]
    out_dir = opts[:out]
    iterations = opts[:iterations] || 100
    batch_size = opts[:batch_size] || 8
    learning_rate = opts[:learning_rate] || 1.0e-4
    log_every = opts[:log] || 50

    if not out_dir or out_dir == "" do
      Mix.raise("--out DIR is required")
    end

    out_dir = resolve(out_dir)

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("checkpoint not found: #{ckpt_dir}")
    end

    unless File.regular?(fixture_path) do
      Mix.raise("fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.regular?(train_path) do
      Mix.raise("train sequences not found: #{train_path}. Run Fetch first.")
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
      item_embeddings =
        if embeddings_path && File.regular?(embeddings_path) do
          Mix.shell().info("Loading embeddings from #{embeddings_path}...")
          RecGPT.Embedding.load_embeddings(embeddings_path)
        else
          unless File.regular?(items_path) do
            Mix.raise("items not found: #{items_path} and no --embeddings given")
          end

          Mix.shell().info("Encoding item text from #{items_path}...")
          raw = File.read!(items_path) |> Jason.decode!()
          items = raw["items"] || []
          n = raw["num_items"] || length(items)

          item_text_dict =
            items
            |> Enum.take(n)
            |> Enum.with_index()
            |> Map.new(fn {item, idx} -> {idx, item["title"] || item["text"] || ""} end)

          RecGPT.Embedding.encode_item_text_dict(item_text_dict)
        end

      stream =
        RecGPT.AxonTrain.stream_batches(sequences, token_id_list, item_embeddings,
          batch_size: batch_size,
          epochs: 1,
          shuffle: true
        )

      Mix.shell().info("Training for up to #{iterations} steps (batch_size=#{batch_size})...")

      trained =
        RecGPT.AxonTrain.run(stream, params,
          iterations: iterations,
          log: log_every,
          learning_rate: learning_rate
        )

      Mix.shell().info("Writing export to #{out_dir}...")
      RecGPT.CheckpointExport.write_export(trained, out_dir)
      Mix.shell().info("Done.")
    end
  end

  defp resolve(path) do
    if absolute_path?(path), do: path, else: Path.expand(path, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
