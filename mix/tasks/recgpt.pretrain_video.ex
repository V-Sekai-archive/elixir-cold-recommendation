defmodule Mix.Tasks.Recgpt.PretrainVideo do
  @shortdoc "Pretrain on video-domain sequences (KuaiRand watch trajectories)"
  @moduledoc """
  Runs pretraining using the **video domain** pipeline: sequences are user watch
  trajectories (e.g. from KuaiRand-Pure). Same shapes and loss as generic pretrain;
  defaults point at `data/kuairand` produced by `mix recgpt.convert_trajectories`.

  Use this to run the sequence pipeline on video-domain data (next-video prediction).
  See [94 Video-domain trajectory test](docs/features/94_video_domain_trajectory_test.md).

  ## Pipeline (run before this task)

      1. mix recgpt.convert_trajectories --from /path/to/KuaiRand-Pure --out data/kuairand --format kuairand
      2. mix recgpt.fetch_vae_ckpt   # FSQ (semantic IDs); embedder downloads on first use
      3a. (optional) mix recgpt.kuairand_canonical_texts --out data/kuairand/item_canonical_texts.json   # full enrichment from video features
      3. mix recgpt.build_fixture --items data/kuairand/items.json --out data/kuairand/fixture.json --ckpt data/fuxi_ckpt_export --vae-ckpt thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt --canonical-texts-from data/kuairand/item_canonical_texts.json
  (Omit 3a and use --no-canonical-texts to use items.json titles only.)

  Then:

      mix recgpt.pretrain_video --out data/kuairand/ckpt_pretrained

  ## Options

    * `--out` - Output checkpoint dir (required)
    * `--data-dir` - Video data directory (default: data/kuairand). Used for fixture, train, items, test paths.
    * `--db` - Use train and items from SQLite (constant memory). Requires prior convert --sync-to-db and build_fixture with DB.
    * `--epochs` - Full passes over training data (default: 5)
    * `--ckpt` - Checkpoint dir (default: data/fuxi_ckpt_export or artifact)
    * `--fixture` - Override fixture path (default: <data-dir>/fixture.json)
    * `--save-every` - Save checkpoint every N steps (0 = disable)
    * `--eval-test-every` - Compute test loss every N steps (default: 0; uses <data-dir>/test_sequences.json when set)
    * `--test` - Override test_sequences path
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50)
    * `--log-interval-sec` - Log at least every N seconds (default: 20)
    * `--limit` - Max items to use (default: fixture num_items)
    * `--mtp-loss-weight` - MTP loss weight (default: 1.0)

  ## Examples

  File-based (default):

      mix recgpt.pretrain_video --out data/kuairand/ckpt_pretrained --epochs 5 --eval-test-every 200

  Constant-memory from DB (after convert --sync-to-db and build_fixture with items db):

      mix recgpt.pretrain_video --out data/kuairand/ckpt_pretrained --db --epochs 5

  Custom data dir:

      mix recgpt.pretrain_video --data-dir /path/to/video_data --out /path/to/video_data/ckpt_pretrained

  Full options (file-based):

      mix recgpt.pretrain_video \\
        --out data/kuairand/ckpt_pretrained \\
        --data-dir data/kuairand \\
        --epochs 5 \\
        --ckpt data/fuxi_ckpt_export \\
        --save-every 500 \\
        --eval-test-every 200 \\
        --batch-size 8 \\
        --learning-rate 1.0e-4 \\
        --log 50 \\
        --log-interval-sec 20 \\
        --limit 10000 \\
        --mtp-loss-weight 1.0
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          out: :string,
          data_dir: :string,
          db: :boolean,
          epochs: :integer,
          ckpt: :string,
          fixture: :string,
          save_every: :integer,
          eval_test_every: :integer,
          test: :string,
          batch_size: :integer,
          learning_rate: :float,
          log: :integer,
          log_interval_sec: :integer,
          limit: :integer,
          mtp_loss_weight: :float
        ]
      )

    Application.ensure_all_started(:recgpt)

    data_dir = opts[:data_dir] || path("data/kuairand")
    out = opts[:out]
    use_db? = opts[:db] == true

    if is_nil(out) or out == "" do
      Mix.raise("--out DIR is required")
    end

    # Build pretrain args with video-domain paths
    train_path = if use_db?, do: "db", else: Path.join(data_dir, "train_sequences.json")
    items_path = if use_db?, do: "db", else: Path.join(data_dir, "items.json")
    fixture_path = opts[:fixture] || Path.join(data_dir, "fixture.json")
    ckpt_dir = opts[:ckpt]
    test_path = opts[:test] || Path.join(data_dir, "test_sequences.json")
    eval_test_every = opts[:eval_test_every] || 0

    pretrain_args = [
      "--out",
      out,
      "--train",
      train_path,
      "--items",
      items_path,
      "--fixture",
      fixture_path,
      "--epochs",
      (opts[:epochs] || 5) |> Integer.to_string()
    ]

    pretrain_args = if ckpt_dir, do: ["--ckpt", ckpt_dir | pretrain_args], else: pretrain_args

    pretrain_args =
      if opts[:save_every],
        do: ["--save-every", to_string(opts[:save_every]) | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if eval_test_every > 0,
        do: ["--eval-test-every", to_string(eval_test_every), "--test", test_path | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if opts[:batch_size],
        do: ["--batch-size", to_string(opts[:batch_size]) | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if opts[:learning_rate],
        do: ["--learning-rate", to_string(opts[:learning_rate]) | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if opts[:log], do: ["--log", to_string(opts[:log]) | pretrain_args], else: pretrain_args

    pretrain_args =
      if opts[:log_interval_sec],
        do: ["--log-interval-sec", to_string(opts[:log_interval_sec]) | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if opts[:limit],
        do: ["--limit", to_string(opts[:limit]) | pretrain_args],
        else: pretrain_args

    pretrain_args =
      if opts[:mtp_loss_weight],
        do: ["--mtp-loss-weight", to_string(opts[:mtp_loss_weight]) | pretrain_args],
        else: pretrain_args

    Mix.Task.run("recgpt.pretrain", pretrain_args)
  end

  defp path(relative) do
    if absolute_path?(relative), do: relative, else: Path.expand(relative, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
