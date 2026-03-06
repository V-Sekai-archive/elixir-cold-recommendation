# RecGPT.AxonTrain: model mirrors Inference, same batch format and loss.
defmodule RecGPT.AxonTrainTest do
  use ExUnit.Case, async: true

  alias RecGPT.AxonTrain
  alias RecGPT.FSQEncoder
  alias RecGPT.Training

  defp dummy_params do
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})

    %{
      "wte" => wte,
      "pred_head.weight" => head_w,
      "pred_head.bias" => head_b
    }
  end

  test "predict returns full-sequence logits (batch, seq_len, 15_361)" do
    params = dummy_params()
    batch = 2
    seq_len = 4
    batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {batch, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})
    input = {batch_token_ids, batch_aux_embeds, embed_mask}

    logits = AxonTrain.predict(params, input)
    assert Nx.shape(logits) == {batch, seq_len, 15_361}
  end

  test "loss_fn matches Training.loss_shifted_ce (Axon passes y_true, y_pred)" do
    batch = 2
    seq_len = 4
    vocab = 15_361
    logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)
    labels = Nx.tensor([[1, 2, 3, -100], [0, 1, -100, -100]], type: {:s, 32})

    loss_axon = AxonTrain.loss_fn(labels, logits)
    loss_training = Training.loss_shifted_ce(logits, labels)
    assert Nx.shape(loss_axon) == {}
    assert Nx.to_number(loss_axon) == Nx.to_number(loss_training)
  end

  test "trainer builds a loop" do
    loop = AxonTrain.trainer()
    assert %Axon.Loop{} = loop
  end

  test "stream_batches yields {inputs, labels} with correct shapes" do
    num_items = 5

    item_embeddings =
      Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items) |> Nx.as_type({:f, 32})

    project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.subtract(0.05)
    project_in_b = Nx.broadcast(0.0, {5})
    project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.subtract(0.05)
    project_out_b = Nx.broadcast(0.0, {192})

    fsq_params = %{
      "project_in" => %{"kernel" => project_in_k, "bias" => project_in_b},
      "project_out" => %{"kernel" => project_out_k, "bias" => project_out_b}
    }

    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(item_embeddings, fsq_params, 2)
    # 4 seqs so first batch is full (batch_size 2)
    seqs = [[0, 1, 2], [1, 2, 3], [0, 3, 4], [2, 4]]

    stream =
      AxonTrain.stream_batches(seqs, token_id_list, item_embeddings,
        batch_size: 2,
        shuffle: false
      )

    [first_batch | _] = Enum.take(stream, 1)

    {{batch_seq, batch_aux, embed_mask}, batch_labels} = first_batch
    assert Nx.shape(batch_seq) == {2, 2048}
    assert Nx.shape(batch_labels) == {2, 2048}
    assert Nx.shape(batch_aux) == {2, 256 * 4, 192}
    assert Nx.shape(embed_mask) == {2, 256 * 4, 1}
  end

  @tag :integration
  test "run runs one iteration with checkpoint params and returns updated flat params" do
    params = dummy_params()
    batch = 2
    seq_len = 8
    batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {batch, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})
    batch_labels = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})

    batch_labels =
      Nx.select(Nx.less(batch_labels, 50), batch_labels, Nx.broadcast(-100, {batch, seq_len}))

    one_batch = [{{batch_token_ids, batch_aux_embeds, embed_mask}, batch_labels}]

    result = AxonTrain.run(one_batch, params, iterations: 1, log: 1)
    assert is_map(result)
    assert Map.has_key?(result, "wte")
    assert Map.has_key?(result, "pred_head.weight")
    # Returned params are still usable for inference (full-sequence forward)
    {input, _labels} = hd(one_batch)
    logits = AxonTrain.predict(result, input)
    assert Nx.shape(logits) == {batch, seq_len, 15_361}
  end

  test "run with optimizer :sgd completes and returns updated params" do
    params = dummy_params()
    batch = 2
    seq_len = 4
    batch_token_ids = Nx.iota({batch, seq_len}) |> Nx.remainder(100) |> Nx.as_type({:s, 32})
    batch_aux_embeds = Nx.broadcast(0.0, {batch, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {batch, seq_len, 1}) |> Nx.as_type({:f, 32})
    batch_labels = Nx.tensor([[1, 2, 3, -100], [0, 1, 2, 3]], type: {:s, 32})
    one_batch = [{{batch_token_ids, batch_aux_embeds, embed_mask}, batch_labels}]

    result = AxonTrain.run(one_batch, params, iterations: 1, optimizer: :sgd, log: 0)
    assert is_map(result)
    assert Map.has_key?(result, "wte")
  end
end
