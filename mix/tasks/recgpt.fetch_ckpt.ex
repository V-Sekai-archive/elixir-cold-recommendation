defmodule Mix.Tasks.Recgpt.FetchCkpt do
  @shortdoc "Download RecGPT checkpoint from Hugging Face (hkuds/RecGPT_model)"
  @moduledoc """
  Downloads the RecGPT PyTorch checkpoint from [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model/tree/main).

  Saves `recgpt_layer_3_weight.pt` (~290 MB) to the given path. Then run
  `mix recgpt.export_ckpt --from-pt path/to/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export`
  to produce the Elixir export (manifest.json + .npy). Load with `RecGPT.CheckpointLoader.load_from_export/1`.

  ## Options
    * `--out` - Output path (default: data/recgpt_layer_3_weight.pt). Use a full path, e.g. `--out thirdparty/RecGPT_model/recgpt_layer_3_weight.pt`.

  ## Examples
      mix recgpt.fetch_ckpt
      mix recgpt.fetch_ckpt --out thirdparty/RecGPT_model/recgpt_layer_3_weight.pt
  """
  use Mix.Task

  @hf_url "https://huggingface.co/hkuds/RecGPT_model/resolve/main/recgpt_layer_3_weight.pt?download=true"
  @filename "recgpt_layer_3_weight.pt"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [out: :string])

    out_path = opts[:out] || Path.join(File.cwd!(), "data/#{@filename}")

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
        {:ok, expected_bytes} ->
          unless valid_pt_file?(out_path) do
            File.rm(out_path)
            Mix.raise(
              "Downloaded file is not a valid .pt (zip format). Hugging Face may have returned HTML. " <>
                "Use a pre-exported checkpoint or run fetch_ckpt from a network that receives the binary."
            )
          end

          if expected_bytes do
            actual = File.stat!(out_path).size
            if actual != expected_bytes do
              File.rm(out_path)
              Mix.raise(
                "Download incomplete: got #{actual} bytes, expected #{expected_bytes}. " <>
                  "Retry or use a pre-exported checkpoint."
              )
            end
          end

          Mix.shell().info("Done. Export to Elixir format (manifest + .npy):")

          Mix.shell().info(
            "  mix recgpt.export_ckpt --from-pt #{out_path} --out data/recgpt_ckpt_export"
          )

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
        {:ok, %{status: 200} = resp} ->
          expected =
            case Req.Response.get_header(resp, "content-length") do
              [len] -> String.to_integer(len)
              _ -> nil
            end
          {:ok, expected}

        {:ok, %{status: code}} ->
          {:error, "HTTP #{code}"}

        {:error, reason} ->
          {:error, reason}
      end
    after
      File.close(file)
    end
  end

  # PyTorch 1.6+ .pt is zip; first 4 bytes are PK\x03\x04
  defp valid_pt_file?(path) do
    fd = File.open!(path, [:read, :binary, :raw])
    first = IO.binread(fd, 4)
    File.close(fd)
    first == <<0x50, 0x4B, 0x03, 0x04>>
  end
end
