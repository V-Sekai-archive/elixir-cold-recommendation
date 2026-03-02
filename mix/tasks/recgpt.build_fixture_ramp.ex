defmodule Mix.Tasks.Recgpt.BuildFixtureRamp do
  @shortdoc "Build fixture with increasing limits (100, 200, ...) until all items or failure"
  @moduledoc """
  Runs build_fixture with slowly increasing --limit (start, start+step, ...) until
  the catalog reaches all items in items.json or a step fails. Use to find the
  maximum catalog size that works on your machine.

  ## Options
    * `--items` - Path to items.json (default: data/steam/items.json)
    * `--out` - Output fixture path (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--start` - First limit to try (default: 100)
    * `--step` - Add this many items each time (default: 100). Next limits: start, start+step, start+2*step, ...
    * `--max` - Stop increasing at this limit (default: use full num_items from items.json)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          items: :string,
          out: :string,
          ckpt: :string,
          start: :integer,
          step: :integer,
          max: :integer
        ]
      )

    items_path = opts[:items] || resolve("data/steam/items.json")
    out_path = opts[:out] || resolve("data/steam/fixture.json")
    ckpt_dir = opts[:ckpt] || resolve("data/recgpt_ckpt_export")
    start_limit = opts[:start] || 100
    step = opts[:step] || 100
    max_limit = opts[:max]

    unless File.regular?(items_path) do
      Mix.raise("items file not found: #{items_path}. Run mix recgpt.fetch_steam first.")
    end

    raw = File.read!(items_path) |> Jason.decode!()
    total_items = raw["num_items"] || length(raw["items"] || [])

    cap =
      if is_integer(max_limit) and max_limit > 0,
        do: min(max_limit, total_items),
        else: total_items

    if start_limit > cap do
      Mix.raise("--start #{start_limit} > available items (#{cap}). Use --max or fewer items.")
    end

    Mix.shell().info("Ramping fixture limit from #{start_limit} toward #{cap} (step #{step})...")

    limits = ramp_limits(start_limit, step, cap)

    last_ok =
      Enum.reduce_while(limits, nil, fn limit, acc ->
        Mix.shell().info("Trying limit #{limit}...")

        argv = [
          "--items",
          items_path,
          "--out",
          out_path,
          "--ckpt",
          ckpt_dir,
          "--limit",
          to_string(limit)
        ]

        case run_build_fixture(argv) do
          :ok ->
            {:cont, limit}

          {:error, msg} ->
            Mix.shell().error("Failed at limit #{limit}: #{msg}")
            {:halt, acc}
        end
      end)

    if last_ok do
      Mix.shell().info(
        "Done. Last successful limit: #{last_ok}. Fixture at #{out_path} has #{last_ok} items."
      )
    else
      Mix.shell().info(
        "Ramp stopped after first failure. Check RECGPT_MAX_MEMORY_MB or NIF limits."
      )
    end
  end

  defp ramp_limits(start, step, cap) when start <= cap do
    Stream.iterate(start, &(&1 + step)) |> Stream.take_while(&(&1 <= cap)) |> Enum.to_list()
  end

  defp run_build_fixture(argv) do
    task = Mix.Task.get("recgpt.build_fixture")

    try do
      task.run(argv)
      :ok
    rescue
      e -> {:error, Exception.message(e)}
    catch
      kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
    end
  end

  defp resolve(path) do
    if String.starts_with?(path, "/") or path =~ ~r/^[a-zA-Z]:/,
      do: path,
      else: Path.expand(path, File.cwd!())
  end
end
