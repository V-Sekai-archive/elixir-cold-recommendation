defmodule Mix.Tasks.Recgpt.FetchCkpt do
  @shortdoc "Download RecGPT checkpoint from Hugging Face (hkuds/RecGPT_model)"
  @moduledoc """
  Downloads the RecGPT PyTorch checkpoint from [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model/tree/main).

  Saves `recgpt_layer_3_weight.pt` (~290 MB) to the given path. Then run
  `mix recgpt.export_ckpt --from-pt path/to/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export`
  to produce the Elixir export (manifest.json + .npy). Load with `RecGPT.CheckpointLoader.load_from_export/1`.

  ## Options
    * `--out` - Output path (default: data/recgpt_layer_3_weight.pt)
    * `--base` - Base directory; file is written as base/recgpt_layer_3_weight.pt (overrides --out if set)

  ## Examples
      mix recgpt.fetch_ckpt
      mix recgpt.fetch_ckpt --out thirdparty/RecGPT_model/recgpt_layer_3_weight.pt
      mix recgpt.fetch_ckpt --base thirdparty/RecGPT_model
  """
  use Mix.Task

  @hf_url "https://huggingface.co/hkuds/RecGPT_model/resolve/main/recgpt_layer_3_weight.pt"
  @filename "recgpt_layer_3_weight.pt"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [out: :string, base: :string])

    out_path =
      if base = opts[:base] do
        Path.join(base, @filename)
      else
        opts[:out] || Path.join(File.cwd!(), "data/#{@filename}")
      end

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
          Mix.shell().info("Done. Export to Elixir format (manifest + .npy):")
          Mix.shell().info("  mix recgpt.export_ckpt --from-pt #{out_path} --out data/recgpt_ckpt_export")

        {:error, reason} ->
          Mix.raise("Download failed: #{inspect(reason)}")
      end
    end
  end

  defp stream_download(url, path) do
    file = File.open!(path, [:write, :binary, :raw])

    try do
      case Req.get(url,
             into: fn {:data, data}, acc ->
               IO.binwrite(file, data)
               {:cont, acc}
             end
           ) do
        {:ok, %{status: 200}} -> :ok
        {:ok, %{status: code}} -> {:error, "HTTP #{code}"}
        {:error, reason} -> {:error, reason}
      end
    after
      File.close(file)
    end
  end
end
