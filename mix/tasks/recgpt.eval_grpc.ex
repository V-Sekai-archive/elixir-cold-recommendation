defmodule Mix.Tasks.Recgpt.EvalGrpc do
  @shortdoc "Evaluate Steam catalogue recommendations via gRPC Predict path"
  @moduledoc """
  Runs evaluation by calling the **gRPC Predict** handler (same code path as
  recgpt.v1.PredictionService/Predict). Loads Steam fixture, checkpoint, and
  optional catalog; sets serve_state; then for each test case from
  test_sequences.json calls the PredictionService.Server and computes Hit@k, MRR.

  Use this to verify that recommendations from Steam's catalogue are evaluated
  through the gRPC API logic. Same metrics as `mix recgpt.eval`; different
  code path (Predict RPC handler instead of direct Serve.recommend).

  ## Options
    * `--data-dir` - Dataset dir (default: data/steam)
    * `--fixture` - Path to fixture JSON. Default: <data-dir>/fixture.json
    * `--ckpt` - Checkpoint export dir.
    * `--test` - Path to test_sequences.json (or cold_test_sequences.json with --cold).
    * `--catalog` - Optional path to items.json (for display_name in responses).
    * `--cold` - Use cold test split.
    * `--top-k` - Top-k for MRR (default: 10)
    * `--progress` - Print progress every N seconds (default: 0 = off)

  ## Example
      mix recgpt.eval_grpc --data-dir data/steam
      mix recgpt.eval_grpc --fixture data/steam/fixture.json --ckpt data/fuxi_ckpt_export --test data/steam/test_sequences.json --catalog data/steam/items.json
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
          catalog: :string,
          cold: :boolean,
          top_k: :integer,
          progress: :integer
        ]
      )

    data_dir = opts[:data_dir] || System.get_env("RECGPT_DATA_DIR") || "data/steam"
    data_dir = Path.expand(data_dir, File.cwd!())

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        RecGPT.Catalog.Artifact.resolve_path("fixture") ||
        Path.join(data_dir, "fixture.json")

    fixture_path = Path.expand(fixture_path, File.cwd!())

    ckpt_dir =
      opts[:ckpt] ||
        System.get_env("RECGPT_CKPT_PATH") ||
        RecGPT.Catalog.Artifact.resolve_path("checkpoint") ||
        Path.join(File.cwd!(), "data/fuxi_ckpt_export")

    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())

    catalog_path =
      (opts[:catalog] && Path.expand(opts[:catalog], File.cwd!())) ||
        RecGPT.Catalog.Artifact.resolve_path("items")

    test_path =
      opts[:test] ||
        RecGPT.Catalog.Artifact.resolve_path(
          if(opts[:cold], do: "cold_test_sequences", else: "test_sequences")
        ) ||
        if(opts[:cold],
          do: Path.join(data_dir, "cold_test_sequences.json"),
          else: Path.join(data_dir, "test_sequences.json")
        )

    test_path = Path.expand(test_path, File.cwd!())

    top_k = opts[:top_k] || 10
    progress_sec = opts[:progress] || 0

    unless File.regular?(fixture_path) do
      Mix.raise("Fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("Checkpoint not found: #{ckpt_dir}. Run mix recgpt.export_ckpt first.")
    end

    Mix.shell().info("Loading state (fixture=#{fixture_path}, ckpt=#{ckpt_dir})...")

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
      {:ok, state} ->
        Application.put_env(:recgpt, :serve_state, state)

        case RecGPT.Eval.load_test_cases(test_path) do
          {:ok, cases} ->
            cases = RecGPT.Eval.filter_to_catalog(cases, state.num_items)
            n = length(cases)

            if n == 0 do
              Mix.shell().info(
                "No test cases in catalog range. Check fixture limit and test file."
              )

              return_ok()
            end

            Mix.shell().info("Evaluating #{n} test cases via gRPC Predict (test=#{test_path})...")

            grpc_recommend_fn = fn ctx, k ->
              request = %Recgpt.V1.PredictRequest{
                context_item_ids: ctx,
                max_results: k
              }

              try do
                response = Recgpt.V1.PredictionService.Server.predict(request, nil)
                {:ok, response.item_ids || []}
              rescue
                e in GRPC.RPCError -> {:error, e}
              end
            end

            eval_opts = [
              top_k: min(top_k, 20),
              total: n,
              recommend_fn: grpc_recommend_fn
            ]

            eval_opts =
              if progress_sec > 0,
                do: Keyword.put(eval_opts, :progress_interval_sec, progress_sec),
                else: eval_opts

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
    Mix.shell().info("Evaluation (gRPC Predict path, Steam catalogue)")
    Mix.shell().info("  n = #{metrics[:n]}")
    Mix.shell().info("  hit_at_1 = #{Float.round(metrics[:hit_at_1] || 0, 4)}")
    Mix.shell().info("  hit_at_5 = #{Float.round(metrics[:hit_at_5] || 0, 4)}")
    Mix.shell().info("  hit_at_10 = #{Float.round(metrics[:hit_at_10] || 0, 4)}")
    Mix.shell().info("  mrr = #{Float.round(metrics[:mrr] || 0, 4)}")
    Mix.shell().info("  catalog_size = #{metrics[:catalog_size]}")
    Mix.shell().info("  random_hit_at_1 = #{Float.round(metrics[:random_hit_at_1] || 0, 4)}")
    Mix.shell().info("  rejects_null = #{metrics[:rejects_null]}")
    if metrics[:halted], do: Mix.shell().info("  (halted: #{inspect(metrics[:halted])})")
  end

  defp return_ok, do: :ok
end
