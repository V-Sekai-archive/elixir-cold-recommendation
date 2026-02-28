defmodule Mix.Tasks.Recgpt.Eval do
  @shortdoc "Run next-item evaluation (Hit@k, MRR) on a standard test set"
  @moduledoc """
  Loads fixture + checkpoint, runs RecGPT.Eval on a JSON test set, prints metrics.

  Use with Steam or other FOSS dataset so numbers are comparable.
  Prepare test data: run `mix recgpt.fetch_steam data/steam`, then `mix recgpt.build_fixture` to get fixture and sequences.

  ## Options
    * `--fixture` - Path to fixture JSON (token_id_list; default: data/steam/fixture.json)
    * `--ckpt` - Path to checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--test` - Path to test_sequences.json (default: data/steam/test_sequences.json)
    * `--cold-test` - Path to cold_test_sequences.json (optional; default: data/steam/cold_test_sequences.json)
    * `--catalog` - Optional catalog JSON (for serve; not required for eval)

  ## Environment
    * RECGPT_FIXTURE, RECGPT_CKPT_EXPORT - override paths

  ## Output
    Prints n, Hit@1, Hit@5, Hit@10, MRR, and random baseline (1/N) so you can compare.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          fixture: :string,
          ckpt: :string,
          test: :string,
          cold_test: :string,
          catalog: :string
        ]
      )

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        resolve_path("data/steam/fixture.json")

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_EXPORT") ||
        resolve_path("data/recgpt_ckpt_export")

    test_path = opts[:test] || resolve_path("data/steam/test_sequences.json")
    cold_test_path = opts[:cold_test] || resolve_path("data/steam/cold_test_sequences.json")
    catalog_path = opts[:catalog]

    Application.ensure_all_started(:nx)

    with {:ok, state} <- RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path),
         {:ok, test_cases} <- RecGPT.Eval.load_test_cases(test_path) do
      metrics = RecGPT.Eval.evaluate(state, test_cases, top_k: 10)

      Mix.shell().info("Evaluation (standard test set)")
      Mix.shell().info("  n = #{metrics.n}")

      Mix.shell().info(
        "  Hit@1  = #{format(metrics.hit_at_1)}  (random baseline #{format(metrics.random_hit_at_1)})"
      )

      Mix.shell().info("  Hit@5  = #{format(metrics.hit_at_5)}")
      Mix.shell().info("  Hit@10 = #{format(metrics.hit_at_10)}")
      Mix.shell().info("  MRR    = #{format(metrics.mrr)}")
      Mix.shell().info("  catalog_size = #{metrics.catalog_size}")

      Mix.shell().info(
        "  Reject null (Hit@1 > random): #{if metrics.rejects_null, do: "yes", else: "no"}"
      )

      if File.regular?(cold_test_path) do
        case RecGPT.Eval.load_test_cases(cold_test_path) do
          {:ok, cold_test_cases} ->
            cold_metrics = RecGPT.Eval.evaluate(state, cold_test_cases, top_k: 10)
            Mix.shell().info("")
            Mix.shell().info("Cold test")
            Mix.shell().info("  n = #{cold_metrics.n}")
            Mix.shell().info(
              "  Hit@1  = #{format(cold_metrics.hit_at_1)}  (random baseline #{format(cold_metrics.random_hit_at_1)})"
            )
            Mix.shell().info("  Hit@5  = #{format(cold_metrics.hit_at_5)}")
            Mix.shell().info("  Hit@10 = #{format(cold_metrics.hit_at_10)}")
            Mix.shell().info("  MRR    = #{format(cold_metrics.mrr)}")
            Mix.shell().info("  catalog_size = #{cold_metrics.catalog_size}")
            Mix.shell().info(
              "  Reject null (Hit@1 > random): #{if cold_metrics.rejects_null, do: "yes", else: "no"}"
            )

          _ ->
            :ok
        end
      end
    else
      {:error, reason} when is_binary(reason) ->
        Mix.raise("Eval failed: #{reason}")

      {:error, reason} ->
        Mix.raise("Eval failed: #{inspect(reason)}")
    end
  end

  defp format(x) when is_float(x), do: :io_lib.format("~.4f", [x]) |> to_string()

  defp resolve_path(path) do
    if absolute_path?(path),
      do: path,
      else:
        first_existing(Path.join(File.cwd!(), path), Path.join(Path.dirname(File.cwd!()), path))
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/

  defp first_existing(a, b) do
    cond do
      File.regular?(a) or File.dir?(a) -> a
      File.regular?(b) or File.dir?(b) -> b
      true -> a
    end
  end
end
