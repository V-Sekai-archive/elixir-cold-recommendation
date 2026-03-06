defmodule Mix.Tasks.Recgpt.KuairandCanonicalTexts do
  @shortdoc "Emit JSON-LD XMP canonicalized text per video from KuaiRand-Pure item features"
  @moduledoc """
  Reads all video item features from thirdparty/KuaiRand-Pure (basic + statistic CSVs),
  joins by video_id, and writes one canonical JSON-LD text per item to a JSON file.

  The output is suitable for embedding: each item's full feature set is a deterministic
  JSON-LD string (sorted keys, XMP/schema.org style). Use as canonical item text or
  item_embedding_text when building fixture or syncing to DB.

  ## Options
    * `--from` - KuaiRand-Pure directory (default: thirdparty/KuaiRand-Pure)
    * `--out` - Output JSON path (default: data/kuairand/item_canonical_texts.json)

  ## Output format
    * `"by_item_id"` - List of canonical JSON-LD strings in item index order (matches items.json id 0,1,2,...).
    * `"by_video_id"` - Object mapping video_id (string) to canonical JSON-LD string.

  ## Example
      mix recgpt.kuairand_canonical_texts
      mix recgpt.kuairand_canonical_texts --from thirdparty/KuaiRand-Pure --out data/kuairand/item_canonical_texts.json
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [from: :string, out: :string])

    Application.ensure_all_started(:recgpt)

    from_dir = opts[:from] || Path.expand("thirdparty/KuaiRand-Pure", File.cwd!())
    out_path = opts[:out] || Path.expand("data/kuairand/item_canonical_texts.json", File.cwd!())

    unless File.dir?(from_dir) do
      Mix.raise("Directory not found: #{from_dir}")
    end

    case RecGPT.KuaiRand.VideoFeaturesJsonld.load_canonical_texts(from_dir) do
      {:ok, by_video} ->
        sorted_ids = by_video |> Map.keys() |> Enum.sort()
        by_item_id = Enum.map(sorted_ids, &Map.fetch!(by_video, &1))
        by_video_id_str = Map.new(by_video, fn {k, v} -> {Integer.to_string(k), v} end)

        out = %{
          "by_item_id" => by_item_id,
          "by_video_id" => by_video_id_str
        }

        File.mkdir_p!(Path.dirname(out_path))
        File.write!(out_path, Jason.encode!(out, pretty: true))

        Mix.shell().info(
          "Wrote #{length(by_item_id)} item canonical JSON-LD texts to #{out_path}"
        )

        :ok

      {:error, reason} ->
        Mix.raise("Failed to load canonical texts: #{inspect(reason)}")
    end
  end
end
