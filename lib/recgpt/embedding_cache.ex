defmodule RecGPT.EmbeddingCache do
  @moduledoc """
  ETS-based cache for precomputed item embeddings and FSQ tokens.

  Loads item embeddings and tokens from Ecto tables into ETS tables at startup
  for O(1) lookup during inference. This avoids recomputing embeddings (MPNet)
  and FSQ quantization on every inference request.

  Tables:
  - `:recgpt_item_embeddings`: {item_id, embedding_tensor (768-dim)}
  - `:recgpt_item_tokens`: {item_id, [t0, t1, t2, t3]}
  """

  @doc """
  Create and populate ETS tables for item embeddings and tokens.

  Returns {:ok, {embeddings_table, tokens_table}} where each is an ETS table reference.
  Returns {:error, reason} if DB query fails.
  """
  def load_from_db(backend \\ {EXLA.Backend, client: :cuda}) do
    import Ecto.Query
    alias RecGPT.Catalog.ItemEmbedding
    alias RecGPT.Catalog.ItemToken
    alias RecGPT.Repo

    # Create ETS tables
    embeddings_table = :ets.new(:recgpt_item_embeddings, [:set, :protected])
    tokens_table = :ets.new(:recgpt_item_tokens, [:set, :protected])

    try do
      # Load embeddings from DB using all/1 instead of stream (no transaction needed)
      ItemEmbedding
      |> order_by(asc: :item_id)
      |> Repo.all()
      |> Enum.each(fn %{item_id: item_id, embedding: bin} ->
        # Convert binary to tensor and transfer to backend
        tensor =
          binary_to_tensor(bin)
          |> Nx.backend_transfer(backend)

        :ets.insert(embeddings_table, {item_id, tensor})
      end)

      # Load tokens from DB using all/1
      ItemToken
      |> order_by(asc: :item_id)
      |> Repo.all()
      |> Enum.each(fn %{item_id: item_id, t0: t0, t1: t1, t2: t2, t3: t3} ->
        tokens = [t0 || 0, t1 || 0, t2 || 0, t3 || 0]
        :ets.insert(tokens_table, {item_id, tokens})
      end)

      {:ok, {embeddings_table, tokens_table}}
    rescue
      e ->
        :ets.delete(embeddings_table)
        :ets.delete(tokens_table)
        {:error, "Failed to load embedding cache from DB: #{inspect(e)}"}
    end
  end

  @doc """
  Get cached embedding tensor for an item_id.
  Returns {tensor, :ok} if found, {:error, :not_found} if missing.
  """
  def get_embedding(table, item_id) do
    case :ets.lookup(table, item_id) do
      [{^item_id, tensor}] -> {:ok, tensor}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  Get cached FSQ tokens [t0, t1, t2, t3] for an item_id.
  Returns {tokens, :ok} if found, {:error, :not_found} if missing.
  """
  def get_tokens(table, item_id) do
    case :ets.lookup(table, item_id) do
      [{^item_id, tokens}] -> {:ok, tokens}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  Delete ETS tables. Call on shutdown or before reloading.
  """
  def cleanup(embeddings_table, tokens_table) do
    :ets.delete(embeddings_table)
    :ets.delete(tokens_table)
    :ok
  end

  # Convert 768-element binary (f32) to Nx tensor
  defp binary_to_tensor(<<bin::binary>>) do
    Nx.from_binary(bin, {:f, 32})
    |> Nx.reshape({768})
  end
end
