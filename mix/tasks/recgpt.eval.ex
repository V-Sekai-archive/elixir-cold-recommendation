defmodule Mix.Tasks.Recgpt.Eval do
  @shortdoc "Run next-item evaluation (Hit@k, MRR) in Elixir"
  @moduledoc """
  Runs evaluation using **Elixir** (RecGPT.Serve + RecGPT.Eval). Loads fixture and checkpoint,
  then evaluates on test_sequences.json (or cold_test_sequences.json with --cold).

  ## Options
    * `--data-dir` - Dataset dir; fixture and test paths default under this. Default: data/steam
    * `--fixture` - Path to fixture JSON. Default: <data-dir>/fixture.json
    * `--ckpt` - Checkpoint export dir. Default: data/recgpt_ckpt_export
    * `--test` - Path to test_sequences.json (or cold_test_sequences.json with --cold). Default: <data-dir>/test_sequences.json
    * `--cold` - Use cold test split (default: false); sets default test to cold_test_sequences.json
    * `--top-k` - Top-k for MRR (default: 10)
    * `--progress` - Print progress every N seconds (default: 0 = off)

  ## Environment
    * RECGPT_DATA_DIR, RECGPT_CKPT_PATH, RECGPT_FIXTURE - override paths

  ## Output
    Prints Hit@1, Hit@5, Hit@10, MRR, catalog size, random baseline, rejects_null.
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
          cold: :boolean,
          top_k: :integer,
          progress: :integer
        ]
      )

    data_dir = opts[:data_dir] || System.get_env("RECGPT_DATA_DIR") || "data/steam"
    data_dir = Path.expand(data_dir, File.cwd!())

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") || Path.join(data_dir, "fixture.json")
    fixture_path = Path.expand(fixture_path, File.cwd!())

    ckpt_dir = opts[:ckpt] || System.get_env("RECGPT_CKPT_PATH") || Path.join([File.cwd!(), "thirdparty", "checkpoints", "recgpt"])
    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())

    test_path =
      opts[:test] ||
        if(opts[:cold],
          do: Path.join(data_dir, "cold_test_sequences.json"),
          else: Path.join(data_dir, "test_sequences.json")
        )
    test_path = Path.expand(test_path, File.cwd!())

    top_k = opts[:top_k] || 10
    progress_sec = opts[:progress] || 0

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    unless File.regular?(fixture_path) do
      Mix.raise("Fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("Checkpoint not found: #{ckpt_dir}. Run mix recgpt.export_ckpt first.")
    end

    Mix.shell().info("Loading state (fixture=#{fixture_path}, ckpt=#{ckpt_dir})...")

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, nil) do
      {:ok, state} ->
        case RecGPT.Eval.load_test_cases(test_path) do
          {:ok, cases} ->
            cases = RecGPT.Eval.filter_to_catalog(cases, state.num_items)
            n = length(cases)

            if n == 0 do
              Mix.shell().info("No test cases in catalog range. Check fixture limit and test file.")
              return_ok()
            end

            Mix.shell().info("Evaluating #{n} test cases (test=#{test_path})...")

            eval_opts = [top_k: min(top_k, 20), total: n]
            eval_opts = if progress_sec > 0, do: Keyword.put(eval_opts, :progress_interval_sec, progress_sec), else: eval_opts

            metrics = RecGPT.Eval.evaluate(state, cases, eval_opts)

            print_metrics(metrics)
            return_ok()

          {:error, reason} ->
            Mix.raise("Failed to load test cases: #{inspect(reason)}")
        end

      {:error, reason} ->
        Mix.raise("Failed to load state: #{inspect(reason)}")
    end
  end

  defp print_metrics(metrics) do
    Mix.shell().info("Evaluation (Elixir RecGPT)")
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

  defp return_ok do
    :ok
  end
end
