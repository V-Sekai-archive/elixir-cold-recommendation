defmodule Mix.Tasks.Recgpt.TrainingSignalTest do
  @shortdoc "Run training signal test: convert (optional) → build_fixture → pretrain → eval (compare zero-shot vs pretrained)"
  @moduledoc """
  Orchestrates the full training signal test pipeline:

  1. Optionally convert raw data (`--convert-from`)
  2. Build fixture
  3. Pretrain (single, 10min, 5epochs, or compare regime)
  4. Eval zero-shot (baseline)
  5. Eval pretrained (+ cold when cold_test_sequences.json exists)
  6. Print comparison

  ## Usage

      mix recgpt.training_signal_test --convert-from /path/to/movielens-20m
      mix recgpt.training_signal_test --data-dir data/training_signal_test
      mix recgpt.training_signal_test --convert-from /path/to/movielens-20m --regime compare
      mix recgpt.training_signal_test --convert-from /path/to/movielens-20m --fuxi --iterations 50

  ## Options
    * `--data-dir` - Dataset dir (default: data/training_signal_test)
    * `--convert-from` - Raw dataset path; runs convert_trajectories first
    * `--train-limit` - Max train sequences (0 = no cap, default: 0)
    * `--test-limit` - Max test cases (0 = no cap, default: 0)
    * `--ckpt` - Base checkpoint (default: data/fuxi_ckpt_export)
    * `--iterations` - Pretrain steps (default: 500 for single; 1200 for 10min regime)
    * `--epochs` - Pretrain epochs (overrides iterations when set)
    * `--regime` - single (default), 10min, 5epochs, or compare
    * `--fixture-limit` - Max items in fixture (default: 5000)
    * `--fuxi` - Use FuXi-Linear init (default). Saves to ckpt_fuxi_*.
    * `--skip-convert` - Skip convert step (data already converted)
    * `--skip-build` - Skip build_fixture (fixture.json exists)
    * `--skip-pretrain` - Skip pretrain (ckpt exists; eval-only)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          data_dir: :string,
          convert_from: :string,
          train_limit: :integer,
          test_limit: :integer,
          ckpt: :string,
          iterations: :integer,
          epochs: :integer,
          regime: :string,
          fixture_limit: :integer,
          fuxi: :boolean,
          skip_convert: :boolean,
          skip_build: :boolean,
          skip_pretrain: :boolean
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    data_dir = opts[:data_dir] || "data/training_signal_test"
    data_dir = Path.expand(data_dir, File.cwd!())
    convert_from = opts[:convert_from]
    train_limit = opts[:train_limit] || 0
    test_limit = opts[:test_limit] || 0
    fixture_limit = opts[:fixture_limit] || 5000
    regime = opts[:regime] || "single"
    fuxi? = opts[:fuxi] != false
    skip_convert = opts[:skip_convert] || false
    skip_build = opts[:skip_build] || false
    skip_pretrain = opts[:skip_pretrain] || false

    ckpt_dir =
      (opts[:ckpt] || Path.join(File.cwd!(), "data/fuxi_ckpt_export"))
      |> Path.expand(File.cwd!())

    ckpt_dir =
      if fuxi? do
        fuxi_init = Path.join(data_dir, "fuxi_init")

        unless File.dir?(fuxi_init) and File.regular?(Path.join(fuxi_init, "manifest.json")) do
          Mix.shell().info("Step 0: Export FuXi-Linear init params...")
          Mix.Task.reenable("recgpt.export_fuxi_ckpt")
          Mix.Task.run("recgpt.export_fuxi_ckpt", ["--out", fuxi_init])
        end

        fuxi_init
      else
        ensure_checkpoint!(ckpt_dir)
        ckpt_dir
      end

    if convert_from && !skip_convert do
      run_convert(Path.expand(convert_from, File.cwd!()), data_dir, train_limit, test_limit)
    end

    unless skip_build do
      run_build_fixture(data_dir, ckpt_dir, fixture_limit)
    end

    Application.put_env(:recgpt, :ckpt_expected_sha256, nil)

    regime_config = regime_config(regime, opts, fuxi?)
    pretrained_dirs = run_pretrain_regime(data_dir, ckpt_dir, regime_config, skip_pretrain, fuxi?)

    zero_shot = run_eval(data_dir, ckpt_dir)
    zero_shot_cold = run_cold_eval(data_dir, ckpt_dir)

    results = [
      {"Zero-shot", zero_shot, zero_shot_cold}
      | Enum.map(pretrained_dirs, fn {label, dir} ->
          {label, run_eval(data_dir, dir), run_cold_eval(data_dir, dir)}
        end)
    ]

    print_comparison(results)
  end

  defp regime_config("single", opts, fuxi?) do
    iterations = opts[:iterations] || 500
    epochs = opts[:epochs]
    out_suffix = if fuxi?, do: "ckpt_fuxi_pretrained", else: "ckpt_pretrained"

    [
      %{label: "Pretrained", out_suffix: out_suffix, iterations: epochs && nil, epochs: epochs}
      |> maybe_put_iterations(iterations)
    ]
  end

  defp regime_config("10min", opts, fuxi?) do
    iterations = opts[:iterations] || 1200
    out_suffix = if fuxi?, do: "ckpt_fuxi_10min", else: "ckpt_10min"
    [%{label: "10min", out_suffix: out_suffix, iterations: iterations, epochs: nil}]
  end

  defp regime_config("5epochs", _opts, fuxi?) do
    out_suffix = if fuxi?, do: "ckpt_fuxi_5epochs", else: "ckpt_5epochs"
    [%{label: "5epochs", out_suffix: out_suffix, iterations: nil, epochs: 5}]
  end

  defp regime_config("compare", _opts, fuxi?) do
    out_10 = if fuxi?, do: "ckpt_fuxi_10min", else: "ckpt_10min"
    out_5 = if fuxi?, do: "ckpt_fuxi_5epochs", else: "ckpt_5epochs"

    [
      %{label: "10min", out_suffix: out_10, iterations: 1200, epochs: nil},
      %{label: "5epochs", out_suffix: out_5, iterations: nil, epochs: 5}
    ]
  end

  defp regime_config(_, opts, fuxi?), do: regime_config("single", opts, fuxi?)

  defp maybe_put_iterations(r, iterations) when is_integer(iterations),
    do: Map.put(r, :iterations, iterations)

  defp maybe_put_iterations(r, _), do: r

  defp run_pretrain_regime(data_dir, ckpt_dir, configs, skip_pretrain, _fuxi?) do
    Enum.flat_map(configs, fn cfg ->
      out_dir = Path.join(data_dir, cfg.out_suffix)

      unless skip_pretrain do
        run_pretrain(data_dir, ckpt_dir, out_dir, cfg)
      end

      [{cfg.label, out_dir}]
    end)
  end

  defp run_convert(from_dir, out_dir, train_limit, test_limit) do
    Mix.shell().info("Step 1: Convert trajectories from #{from_dir}...")

    case RecGPT.Trajectories.Convert.run(from_dir, out_dir,
           format: :movielens,
           train_limit: train_limit,
           test_limit: test_limit,
           seed: 42
         ) do
      :ok ->
        :ok

      {:error, reason} ->
        Mix.raise("Convert failed: #{inspect(reason)}")
    end
  end

  defp run_build_fixture(data_dir, ckpt_dir, fixture_limit) do
    Mix.shell().info("Step 2: Build fixture...")
    items_path = Path.join(data_dir, "items.json")
    fixture_path = Path.join(data_dir, "fixture.json")

    unless File.regular?(items_path) do
      Mix.raise("items.json not found at #{items_path}. Run with --convert-from first.")
    end

    args = [
      "--items",
      items_path,
      "--out",
      fixture_path,
      "--no-canonical-texts",
      "--ckpt",
      ckpt_dir,
      "--limit",
      to_string(fixture_limit)
    ]

    Mix.Task.reenable("recgpt.build_fixture")
    Mix.Task.run("recgpt.build_fixture", args)
  end

  defp run_pretrain(data_dir, ckpt_dir, out_dir, cfg) do
    Mix.shell().info("Step 3: Pretrain (#{cfg.label})...")
    fixture_path = Path.join(data_dir, "fixture.json")
    train_path = Path.join(data_dir, "train_sequences.json")
    items_path = Path.join(data_dir, "items.json")

    unless File.regular?(fixture_path) do
      Mix.raise("fixture.json not found. Run without --skip-build first.")
    end

    unless File.regular?(train_path) do
      Mix.raise("train_sequences.json not found at #{train_path}.")
    end

    args = [
      "--ckpt",
      ckpt_dir,
      "--fixture",
      fixture_path,
      "--train",
      train_path,
      "--items",
      items_path,
      "--out",
      out_dir
    ]

    args =
      if cfg.epochs do
        args ++ ["--epochs", to_string(cfg.epochs)]
      else
        iterations = cfg.iterations || 500
        args ++ ["--iterations", to_string(iterations)]
      end

    Mix.Task.reenable("recgpt.pretrain")
    Mix.Task.run("recgpt.pretrain", args)
  end

  defp run_eval(data_dir, ckpt_dir) do
    fixture_path = Path.join(data_dir, "fixture.json")
    test_path = Path.join(data_dir, "test_sequences.json")

    if File.regular?(fixture_path) and
         File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) and
         File.regular?(test_path) do
      eval_state(fixture_path, ckpt_dir, test_path)
    else
      nil
    end
  end

  defp run_cold_eval(data_dir, ckpt_dir) do
    cold_path = Path.join(data_dir, "cold_test_sequences.json")
    fixture_path = Path.join(data_dir, "fixture.json")

    if File.regular?(cold_path) and File.regular?(fixture_path) and
         File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      eval_state(fixture_path, ckpt_dir, cold_path)
    else
      nil
    end
  end

  defp eval_state(fixture_path, ckpt_dir, test_path) do
    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, nil) do
      {:ok, state} ->
        case RecGPT.Eval.load_test_cases(test_path) do
          {:ok, cases} ->
            cases = RecGPT.Eval.filter_to_catalog(cases, state.num_items)
            n = length(cases)

            if n == 0 do
              %{n: 0}
            else
              RecGPT.Eval.evaluate(state, cases, top_k: 10, total: n)
            end

          {:error, _} ->
            nil
        end

      {:error, _} ->
        nil
    end
  end

  defp ensure_checkpoint!(ckpt_dir) do
    manifest = Path.join(ckpt_dir, "manifest.json")

    unless File.dir?(ckpt_dir) and File.regular?(manifest) do
      Mix.raise(
        "Checkpoint required at #{ckpt_dir}. Run: mix recgpt.refetch or mix recgpt.export_fuxi_ckpt --out #{ckpt_dir}."
      )
    end
  end

  defp print_comparison(results) do
    Mix.shell().info("")
    Mix.shell().info("=== Training Signal Test ===")

    for {label, metrics, cold_metrics} <- results do
      Mix.shell().info("")
      Mix.shell().info("#{label}:")

      if metrics && metrics[:n] && metrics[:n] > 0 do
        Mix.shell().info("  hit_at_1 = #{Float.round(metrics[:hit_at_1] || 0, 4)}")
        Mix.shell().info("  hit_at_5 = #{Float.round(metrics[:hit_at_5] || 0, 4)}")
        Mix.shell().info("  mrr = #{Float.round(metrics[:mrr] || 0, 4)}")
      else
        Mix.shell().info("  (no test cases or failed to load)")
      end

      if cold_metrics && cold_metrics[:n] && cold_metrics[:n] > 0 do
        Mix.shell().info(
          "  cold: hit_at_1 = #{Float.round(cold_metrics[:hit_at_1] || 0, 4)}, hit_at_5 = #{Float.round(cold_metrics[:hit_at_5] || 0, 4)}"
        )
      end
    end

    zero_shot = List.keyfind(results, "Zero-shot", 0)
    pretrained = Enum.find(results, fn {label, _, _} -> label != "Zero-shot" end)

    if zero_shot && pretrained do
      {_, zs, _} = zero_shot
      {label, pt, _} = pretrained

      if zs && zs[:n] && zs[:n] > 0 && pt && pt[:n] && pt[:n] > 0 do
        improved = (pt[:hit_at_1] || 0) >= (zs[:hit_at_1] || 0)
        Mix.shell().info("")

        Mix.shell().info(
          if improved,
            do: "Signal: #{label} Hit@1 >= zero-shot",
            else: "Signal: #{label} Hit@1 < zero-shot"
        )
      end
    end

    Mix.shell().info("")
  end
end
