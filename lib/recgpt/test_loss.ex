defmodule RecGPT.TestLoss do
  @moduledoc """
  Compute cross-entropy loss on a held-out test set (no gradients).

  Used to monitor generalization during pretraining. Load test_sequences.json,
  convert to sequences (context ++ [next_item]), build batches with Training,
  run forward and loss_shifted_ce, return mean loss.
  """

  alias RecGPT.AxonTrain
  alias RecGPT.Eval
  alias RecGPT.Training

  @doc """
  Computes mean cross-entropy loss on the test set.

  - `params` - Model params (from CheckpointLoader)
  - `token_id_list` - From fixture (list of 4-token lists per item)
  - `item_embeddings` - Tensor (num_items, 768) from Embedding
  - `test_path` - Path to test_sequences.json
  - `opts` - `:batch_size` (default 16), `:limit` (max test cases, nil = all)

  Returns `{:ok, mean_loss}` or `{:error, reason}`.
  """
  @spec compute(map(), [[non_neg_integer()]], Nx.Tensor.t(), String.t(), keyword()) ::
          {:ok, float()} | {:error, term()}
  def compute(params, token_id_list, item_embeddings, test_path, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 16)
    limit = Keyword.get(opts, :limit)

    case Eval.load_test_cases(test_path) do
      {:ok, cases} ->
        cases = if limit, do: Enum.take(cases, limit), else: cases

        sequences =
          cases
          |> Enum.filter(fn c -> (c["context"] || []) != [] or c["next_item"] != nil end)
          |> Enum.map(fn c ->
            ctx = List.wrap(c["context"] || [])
            next = c["next_item"]
            ctx ++ [next]
          end)
          |> Enum.filter(fn seq -> length(seq) >= 1 end)

        if sequences == [] do
          {:error, :no_valid_sequences}
        else
          indices = 0..(length(sequences) - 1)//1 |> Enum.to_list()

          batch_indices_list =
            indices
            |> Enum.chunk_every(batch_size)

          eval_fn =
            Nx.Defn.jit(fn params, input, labels ->
              logits = AxonTrain.predict(params, input)
              AxonTrain.loss_fn(labels, logits)
            end)

          losses =
            Enum.map(batch_indices_list, fn batch_indices ->
              {batch_seq, batch_labels, batch_aux_embeds, embed_mask, _all_timestamps} =
                Training.build_train_batch(sequences, token_id_list, item_embeddings, batch_indices)

              input = {batch_seq, batch_aux_embeds, embed_mask}
              eval_fn.(params, input, batch_labels) |> Nx.to_number()
            end)

          valid_losses = Enum.reject(losses, &(not is_number(&1) or &1 != &1 or &1 == :infinity or &1 == :neg_infinity))
          mean_loss = if valid_losses == [], do: :nan, else: Enum.sum(valid_losses) / length(valid_losses)
          {:ok, mean_loss}
        end

      {:error, _} = err ->
        err
    end
  end
end
