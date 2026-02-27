defmodule Mix.Tasks.Recgpt.Eval do
  @shortdoc "Run next-item evaluation (Hit@k, MRR) on a standard test set"
  @moduledoc """
  Loads fixture + checkpoint, runs RecGPT.Eval on a JSON test set, prints metrics.

  Use with a standard FOSS dataset (e.g. UCI Clickstream) so numbers are comparable.
  Prepare test data: from test env run `RecGPT.Clickstream.Fetch.run()` to get data/clickstream/items.json and test_sequences.json; build fixture from items (Embedding + FSQ).

  ## Options
    * `--fixture` - Path to fixture JSON (token_id_list; default: data/clickstream/fixture.json)
    * `--ckpt` - Path to checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--test` - Path to test_sequences.json (default: data/clickstream/test_sequences.json)
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
        switches: [fixture: :string, ckpt: :string, test: :string, catalog: :string]
      )

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        resolve_path("data/clickstream/fixture.json")

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_EXPORT") ||
        resolve_path("data/recgpt_ckpt_export")

    test_path = opts[:test] || resolve_path("data/clickstream/test_sequences.json")
    catalog_path = opts[:catalog]

    Application.ensure_all_started(:nx)

    with {:ok, state} <- RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path),
         {:ok, test_cases} <- RecGPT.Eval.load_test_cases(test_path) do
      metrics = RecGPT.Eval.evaluate(state, test_cases, top_k: 10)

      Mix.shell().info("Evaluation (standard test set)")
      Mix.shell().info("  n = #{metrics.n}")
      Mix.shell().info("  Hit@1  = #{format(metrics.hit_at_1)}  (random baseline #{format(metrics.random_hit_at_1)})")
      Mix.shell().info("  Hit@5  = #{format(metrics.hit_at_5)}")
      Mix.shell().info("  Hit@10 = #{format(metrics.hit_at_10)}")
      Mix.shell().info("  MRR    = #{format(metrics.mrr)}")
      Mix.shell().info("  catalog_size = #{metrics.catalog_size}")
      Mix.shell().info("  Reject null (Hit@1 > random): #{if metrics.rejects_null, do: "yes", else: "no"}")
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
