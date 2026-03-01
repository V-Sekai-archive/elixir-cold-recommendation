defmodule RecGPT.Embedding do
  @moduledoc """
  Text → 768-d embeddings for RecGPT training (MPNet, sentence-transformers/all-mpnet-base-v2).

  Uses Bumblebee to load and run the model. Mean pooling, no L2 norm (match Python normalize_embeddings=False).
  Outputs are not guaranteed identical to Python; validate if you need parity.

  ## Usage

      serving = RecGPT.Embedding.serving()
      result = Nx.Serving.run(serving, "Some market title Yes")
      result = Nx.Serving.run(serving, ["Text A", "Text B"])
      embeddings = RecGPT.Embedding.encode_item_text_dict(%{0 => "Title A", 1 => "Title B"})
  """

  @model_id "sentence-transformers/all-mpnet-base-v2"
  @embed_batch_size 100

  @doc "Loads the MPNet model and tokenizer, returns a text embedding serving. Cached in application env :recgpt."
  def serving do
    case Application.get_env(:recgpt, :embedding_serving) do
      nil ->
        serving = load_serving!()
        Application.put_env(:recgpt, :embedding_serving, serving)
        serving

      cached ->
        cached
    end
  end

  defp load_serving! do
    IO.puts("Downloading model #{@model_id} (first run may take several minutes)...")

    # Load as :base for hidden_state/pooled_state (sentence-transformers has LM head; we use encoder only).
    {:ok, model_info} =
      Bumblebee.load_model({:hf, @model_id}, spec_overrides: [architecture: :base])

    IO.puts("Model loaded. Downloading tokenizer...")
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model_id})
    IO.puts("Tokenizer loaded. Building serving...")

    serving =
      Bumblebee.Text.text_embedding(model_info, tokenizer,
        output_pool: :mean_pooling,
        output_attribute: :hidden_state,
        embedding_processor: nil
      )

    IO.puts("Embedding serving ready.")
    serving
  end

  defp encode_texts(texts) when is_list(texts) do
    serv = serving()
    results = Nx.Serving.run(serv, texts)
    tensors = Enum.map(results, fn %{embedding: t} -> t end)
    Nx.stack(tensors)
  end

  @doc "Encodes item_text_dict (map of item_index => text) to Nx tensor {num_items, 768}. Indices 0..num_items-1, sorted. Processes in batches of #{@embed_batch_size} to limit memory."
  def encode_item_text_dict(item_text_dict) when is_map(item_text_dict) do
    indices = item_text_dict |> Map.keys() |> Enum.sort()
    texts = Enum.map(indices, &Map.fetch!(item_text_dict, &1))
    encode_texts_batched(texts, @embed_batch_size)
  end

  defp encode_texts_batched(texts, batch_size) when length(texts) <= batch_size do
    encode_texts(texts)
  end

  defp encode_texts_batched(texts, batch_size) do
    texts
    |> Enum.chunk_every(batch_size)
    |> Enum.map(&encode_texts/1)
    |> Nx.concatenate(axis: 0)
  end
end
