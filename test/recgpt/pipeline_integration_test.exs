# Full data pipeline: embeddings -> token_id_list -> train batch -> loss.
# Proves recgpt works end-to-end without HuggingFace model.
defmodule RecGPT.PipelineIntegrationTest do
  use ExUnit.Case, async: true

  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.Training

  defp fsq_params do
    project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.subtract(0.05)
    project_in_b = Nx.broadcast(0.0, {5})
    project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.subtract(0.05)
    project_out_b = Nx.broadcast(0.0, {192})

    %{
      "project_in" => %{"kernel" => project_in_k, "bias" => project_in_b},
      "project_out" => %{"kernel" => project_out_k, "bias" => project_out_b}
    }
  end

  @tag :integration
  test "full pipeline: synthetic embeddings -> token_id_list -> build_train_batch -> loss_shifted_ce" do
    num_items = 5
    # 1. Synthetic item embeddings (num_items, 768)
    item_embeddings =
      Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items) |> Nx.as_type({:f, 32})

    # 2. FSQ params and token_id_list
    params = fsq_params()
    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(item_embeddings, params, 2)
    assert length(token_id_list) == num_items
    assert Enum.all?(token_id_list, fn list -> length(list) == 4 end)

    assert Enum.all?(token_id_list, fn list ->
             Enum.all?(list, fn id -> id >= 0 and id < FSQ.vocab_size() end)
           end)

    # 3. Training sequences (item indices) and batch
    seqs = [[0, 1, 2], [1, 2, 3], [0, 3, 4]]
    batch_indices = [0, 1, 2]

    {batch_seq, batch_labels, batch_aux, embed_mask} =
      Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

    assert Nx.shape(batch_seq) == {3, 1024}
    assert Nx.shape(batch_labels) == {3, 1024}
    assert Nx.shape(batch_aux) == {3, 256 * 4, 192}
    assert Nx.shape(embed_mask) == {3, 256 * 4, 1}

    # 4. Loss with dummy logits (proves batch is consumable)
    vocab = FSQ.vocab_size() + 1
    logits = Nx.iota({3, 1024, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)
    loss = Training.loss_shifted_ce(logits, batch_labels)
    assert Nx.shape(loss) == {}
    loss_val = Nx.to_number(loss)
    assert loss_val >= 0 and loss_val == loss_val
  end

  @tag :integration
  test "pipeline with single sequence and one item" do
    num_items = 1
    item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768) |> Nx.as_type({:f, 32})
    params = fsq_params()
    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(item_embeddings, params)
    assert length(token_id_list) == 1

    seqs = [[0]]

    {batch_seq, batch_labels, batch_aux, _mask} =
      Training.build_train_batch(seqs, token_id_list, item_embeddings, [0])

    assert Nx.shape(batch_seq) == {1, 1024}
    assert Nx.shape(batch_labels) == {1, 1024}
    assert Nx.shape(batch_aux) == {1, 256 * 4, 192}

    # First 4 tokens = item 0's tokens; rest padding
    first_tokens =
      batch_seq
      |> Nx.squeeze(axes: [0])
      |> Nx.slice_along_axis(0, 4, axis: 0)
      |> Nx.to_flat_list()

    assert length(first_tokens) == 4
    assert Enum.all?(first_tokens, fn id -> id >= 0 and id < FSQ.vocab_size() end)
  end
end
