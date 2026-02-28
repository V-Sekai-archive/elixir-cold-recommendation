defmodule RecGPT.FixtureBuild do
  @moduledoc """
  Build fixture.json from items.json: Embedding.encode_item_text_dict + FSQ → token_id_list.

  Used by `mix recgpt.build_fixture`. Output format matches Serve.load_fixture (num_items, token_id_list).
  """

  alias RecGPT.CheckpointLoader
  alias RecGPT.Embedding
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder

  @doc """
  Builds fixture from items path and checkpoint (and optional separate FSQ export).

  - Reads items.json → item_text_dict (id => title for 0..num_items-1).
  - Encodes via Embedding.encode_item_text_dict/1 → {num_items, 768}.
  - Loads FSQ params from ckpt_dir (or fsq_dir if FSQ keys missing in ckpt).
  - Encodes embeddings to token_id_list via FSQEncoder.encode_embeddings_to_token_id_list/3.
  - Returns %{"num_items" => n, "token_id_list" => token_id_list}.

  Options:
  - :fsq_dir - required if checkpoint does not contain FSQ params (project_in/kernel or fsq.project_in.weight, etc.)
  """
  def build(items_path, ckpt_dir, opts \\ []) do
    item_text_dict = load_item_text_dict(items_path)
    embeddings = Embedding.encode_item_text_dict(item_text_dict)
    {num_items, _} = Nx.shape(embeddings)
    num_items = if is_tuple(num_items), do: elem(num_items, 0), else: num_items
    fsq_params = load_fsq_params(ckpt_dir, Keyword.get(opts, :fsq_dir))
    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)
    %{"num_items" => num_items, "token_id_list" => token_id_list}
  end

  @doc "Writes fixture map to path (JSON)."
  def write_fixture(fixture, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(fixture, pretty: true))
    :ok
  end

  defp load_item_text_dict(path) do
    raw = File.read!(path) |> Jason.decode!()
    items = raw["items"] || []
    num_items = raw["num_items"] || length(items)

    items
    |> Enum.take(num_items)
    |> Enum.with_index()
    |> Map.new(fn {item, idx} ->
      title = item["title"] || item["text"] || item["raw"] || ""
      {idx, title}
    end)
  end

  defp load_fsq_params(ckpt_dir, fsq_dir) do
    ckpt_params = CheckpointLoader.load_from_export(ckpt_dir)
    params = FSQ.load_params(ckpt_params)

    cond do
      fsq_params_ok?(params) -> params
      fsq_dir -> load_fsq_params_from_dir(fsq_dir)
      true -> raise_fsq_not_found(ckpt_dir)
    end
  end

  defp load_fsq_params_from_dir(fsq_dir) do
    fsq_export = CheckpointLoader.load_from_export(fsq_dir)
    params_fsq = FSQ.load_params(fsq_export)
    if fsq_params_ok?(params_fsq), do: params_fsq, else: raise_fsq_missing(fsq_dir)
  end

  defp raise_fsq_not_found(ckpt_dir) do
    raise "FSQ params not found in checkpoint #{ckpt_dir}. " <>
            "Export FSQ weights to a directory and pass --fsq <dir>, or use a checkpoint that includes FSQ " <>
            "(e.g. project_in/kernel or fsq.project_in.weight)."
  end

  defp fsq_params_ok?(%{"project_in" => %{"kernel" => k}, "project_out" => %{"kernel" => o}})
       when not is_nil(k) and not is_nil(o), do: true

  defp fsq_params_ok?(_), do: false

  defp raise_fsq_missing(dir) do
    raise "FSQ params still missing after loading from #{dir}. " <>
            "Expected keys: project_in/kernel (or fsq.project_in.weight), project_out/kernel (or fsq.project_out.weight)."
  end
end
