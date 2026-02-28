defmodule RecGPT.FSQEncoder do
  @moduledoc """
  RecGPT FSQ encoder: item embeddings (768-d) to 4 FSQ token IDs per item.
  Requires FSQ params from RecGPT.FSQ.load_params/1.
  """

  alias RecGPT.FSQ

  def encode_embeddings_to_token_id_list(embeddings, fsq_params, batch_size \\ 4096) do
    {num_items, _} = Nx.shape(embeddings)
    num_items = if is_tuple(num_items), do: elem(num_items, 0), else: num_items
    starts = 0..max(0, div(num_items - 1, batch_size)) |> Enum.map(&(&1 * batch_size))

    Enum.reduce(starts, [], fn start, acc ->
      count = min(batch_size, num_items - start)

      if count <= 0 do
        acc
      else
        batch = Nx.slice(embeddings, [start, 0], [count, 768])
        batch_4_192 = Nx.reshape(batch, {count, 4, 192})
        {_quant_embeds, indices} = FSQ.encode(batch_4_192, fsq_params)
        chunks = Nx.to_flat_list(indices) |> Enum.chunk_every(4)
        acc ++ chunks
      end
    end)
  end
end
