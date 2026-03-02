defmodule Mix.Tasks.Recgpt.FetchVaeCkpt do
  @shortdoc "Download VAE checkpoint from HKUDS/RecGPT (for Python eval/predict)"
  @moduledoc """
  Downloads the VAE checkpoint (vae_len4_fsq88865_ep90.pt) from the
  [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) repository so build_fixture
  (--vae-ckpt) and the optional Python pipeline can run. Saves to
  thirdparty/checkpoints/vae/ by default.

  ## Options
    * `--out` - Output path (default: thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt)

  ## Examples
      mix recgpt.fetch_vae_ckpt
      mix recgpt.fetch_vae_ckpt --out path/to/vae.pt
  """
  use Mix.Task

  @vae_url "https://github.com/HKUDS/RecGPT/raw/main/vae_ckpt/vae_len4_fsq88865_ep90.pt"
  @default_filename "vae_len4_fsq88865_ep90.pt"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [out: :string])

    out_path =
      opts[:out] ||
        Path.join([File.cwd!(), "thirdparty", "checkpoints", "vae", @default_filename])

    out_path = Path.expand(out_path)

    Application.ensure_all_started(:req)
    File.mkdir_p!(Path.dirname(out_path))

    if File.regular?(out_path) do
      Mix.shell().info("File already exists: #{out_path}")
      Mix.shell().info("Delete it first to re-download.")
    else
      Mix.shell().info("Downloading VAE checkpoint from HKUDS/RecGPT...")
      Mix.shell().info("  -> #{out_path}")

      case stream_download(@vae_url, out_path) do
        :ok ->
          Mix.shell().info(
            "Done. Use as default (no --vae-ckpt) or set RECGPT_VAE_CKPT=#{out_path}"
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
      receive_timeout: 300_000,
      headers: [
        {"user-agent", "Req/1.0 (Elixir; recgpt fetch_vae_ckpt)"},
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
