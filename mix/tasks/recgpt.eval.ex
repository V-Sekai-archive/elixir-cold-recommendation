defmodule Mix.Tasks.Recgpt.Eval do
  @shortdoc "Run next-item evaluation (Hit@k, MRR) in Elixir"
  @moduledoc """
  Runs evaluation using RecGPT.Serve + RecGPT.Eval. Loads fixture and checkpoint,
  then evaluates on test_sequences.json.

  ## Options
    * `--data-dir` - Dataset dir. Default: data/steam
    * `--fixture` - Path to fixture JSON. Default: <data-dir>/fixture.json
    * `--ckpt` - Checkpoint export dir. Default: data/fuxi_ckpt_export.
    * `--test` - Path to test_sequences.json. Default: <data-dir>/test_sequences.json
    * `--top-k` - Top-k for MRR (default: 10)
    * `--progress` - Print progress every N seconds (default: 0 = off)
    * `--batch-size` - Number of test cases per batched recommend (default: 8)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          data_dir: :string,
          fixture: :string,
          ckpt: :string,
          test: :string,
          top_k: :integer,
          progress: :integer,
          batch_size: :integer
        ]
      )

    data_dir = opts[:data_dir] || System.get_env("RECGPT_DATA_DIR") || "data/steam"
    data_dir = Path.expand(data_dir, File.cwd!())

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        Path.join(data_dir, "fixture.json")

    fixture_path = Path.expand(fixture_path, File.cwd!())

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_PATH") ||
        Path.join([File.cwd!(), "data", "fuxi_ckpt_export"])

    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())

    test_source =
      opts[:test] ||
        Path.join(data_dir, "test_sequences.json")

    test_source = Path.expand(test_source, File.cwd!())

    top_k = opts[:top_k] || 10
    progress_sec = opts[:progress] || 0
    batch_size = opts[:batch_size] || 8

    unless File.regular?(fixture_path) do
      Mix.raise("Fixture not found: #{fixture_path}")
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("Checkpoint not found: #{ckpt_dir}. Run mix recgpt.export_fuxi_ckpt --out #{ckpt_dir}.")
    end

    Mix.shell().info("Loading state (fixture=#{fixture_path}, ckpt=#{ckpt_dir})...")

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir) do
      {:ok, state} ->
        {cases, n} =
          case RecGPT.Eval.load_test_cases(test_source) do
            {:ok, loaded} ->
              cases = RecGPT.Eval.filter_to_catalog(loaded, state.num_items)
              {cases, length(cases)}

            {:error, reason} ->
              Mix.raise("Failed to load test cases: #{inspect(reason)}")
          end

        if n == 0 do
          Mix.shell().info("No test cases in catalog range.")
          :ok
        else
          Mix.shell().info("Evaluating #{n} test cases...")

          eval_opts = [top_k: min(top_k, 20), total: n, batch_size: max(batch_size, 1)]

          eval_opts =
            if progress_sec > 0,
              do: Keyword.put(eval_opts, :progress_interval_sec, progress_sec),
              else: eval_opts

          metrics = RecGPT.Eval.evaluate(state, cases, eval_opts)

          print_metrics(metrics)
          :ok
        end

      {:error, reason} ->
        Mix.raise("Failed to load state: #{inspect(reason)}")
    end
  end

  defp print_metrics(metrics) do
    Mix.shell().info("Evaluation")
    Mix.shell().info("  n = #{metrics[:n]}")
    Mix.shell().info("  hit_at_1 = #{Float.round(metrics[:hit_at_1] || 0, 4)}")
    Mix.shell().info("  hit_at_5 = #{Float.round(metrics[:hit_at_5] || 0, 4)}")
    Mix.shell().info("  hit_at_10 = #{Float.round(metrics[:hit_at_10] || 0, 4)}")
    Mix.shell().info("  mrr = #{Float.round(metrics[:mrr] || 0, 4)}")
    Mix.shell().info("  catalog_size = #{metrics[:catalog_size]}")
    Mix.shell().info("  random_hit_at_1 = #{Float.round(metrics[:random_hit_at_1] || 0, 4)}")
    Mix.shell().info("  rejects_null = #{metrics[:rejects_null]}")

    if metrics[:halted_reason] do
      Mix.shell().info("  (halted: #{inspect(metrics[:halted_reason])})")
    end
  end
end
