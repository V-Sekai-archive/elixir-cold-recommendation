defmodule Mix.Tasks.Recgpt.FetchCkpt do
  @shortdoc "Download RecGPT checkpoint from Hugging Face (hkuds/RecGPT_model)"
  @moduledoc """
  Downloads the RecGPT PyTorch checkpoint from [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model/tree/main).

  Saves `recgpt_layer_3_weight.pt` (~290 MB) to the given path. Then run
  `mix recgpt.export_ckpt --from-pt <path> --out <export_dir>` to produce the
  Elixir export (manifest.json + .npy). Default layout uses thirdparty/checkpoints/recgpt/.

  ## Options
    * `--out` - Output path for the .pt file (default: thirdparty/checkpoints/recgpt/recgpt_layer_3_weight.pt)

  ## Examples
      mix recgpt.fetch_ckpt
      mix recgpt.export_ckpt --from-pt thirdparty/checkpoints/recgpt/recgpt_layer_3_weight.pt --out thirdparty/checkpoints/recgpt
      mix recgpt.fetch_ckpt --out data/recgpt_layer_3_weight.pt
  """
  use Mix.Task

  @hf_url "https://huggingface.co/hkuds/RecGPT_model/resolve/main/recgpt_layer_3_weight.pt?download=true"
  @filename "recgpt_layer_3_weight.pt"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [out: :string])

    out_path =
      opts[:out] ||
        Path.join([File.cwd!(), "thirdparty", "checkpoints", "recgpt", @filename])

    Application.ensure_all_started(:req)
    File.mkdir_p!(Path.dirname(out_path))

    if File.regular?(out_path) do
      Mix.shell().info("File already exists: #{out_path}")
      Mix.shell().info("Delete it first to re-download.")
    else
      Mix.shell().info("Downloading RecGPT checkpoint from Hugging Face (hkuds/RecGPT_model)...")
      Mix.shell().info("  #{@hf_url}")
      Mix.shell().info("  -> #{out_path}")

      case stream_download(@hf_url, out_path) do
        :ok ->
          export_dir = Path.dirname(out_path)
          Mix.shell().info("Done. Export to Elixir format (manifest + .npy):")
          Mix.shell().info("  mix recgpt.export_ckpt --from-pt #{out_path} --out #{export_dir}")

        {:error, reason} ->
          Mix.raise("Download failed: #{inspect(reason)}")
      end
    end
  end

  defp stream_download(url, path) do
    file = File.open!(path, [:write, :binary, :raw])

    opts = [
      into: fn {:data, data}, acc ->
        IO.binwrite(file, data)
        {:cont, acc}
      end,
      max_redirects: 10,
      receive_timeout: 600_000,
      headers: [
        {"user-agent", "Req/1.0 (Elixir; recgpt fetch_ckpt)"},
        {"accept", "application/octet-stream"}
      ]
    ]

    try do
      case Req.get(url, opts) do
        {:ok, %{status: 200}} -> :ok
        {:ok, %{status: code}} -> {:error, "HTTP #{code}"}
        {:error, reason} -> {:error, reason}
      end
    after
      File.close(file)
    end
  end
end
