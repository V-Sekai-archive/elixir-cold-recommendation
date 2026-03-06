defmodule RecGPT.AxonTrain do
  @moduledoc """
  RecGPT training via Axon: model that mirrors Inference (embed, aux, GPT-2 blocks, head),
  checkpoint loaded into params, same batch format and loss as Training.

  Uses Inference.forward_full_sequence for the forward pass and Training.loss_shifted_ce for loss.
  Run training with `run/3`: pass a stream of batches and initial params (flat checkpoint from
  CheckpointLoader.load_from_export/1). Uses [Polaris](https://github.com/elixir-nx/polaris) for
  optimizers (e.g. Adam, SGD) and `Polaris.Updates.apply_updates/2`, with Nx.Defn.value_and_grad.

  Batch format: inputs = {batch_token_ids, batch_aux_embeds, embed_mask}, labels = batch_labels.
  Stream entries: {{batch_token_ids, batch_aux_embeds, embed_mask}, batch_labels}.
  """

  alias RecGPT.FuxiLinearInference
  alias RecGPT.Inference
  alias RecGPT.Training

  @doc """
  Builds the lowered model {init_fn, forward_fn} for Axon.Loop.trainer.

  Note: Axon.Loop.trainer expects Axon.ModelState; flat checkpoint params are not compatible.
  Use `run/3` for training with a loaded checkpoint (flat params); it uses the same forward
  and loss with a custom loop and Polaris optimizer.
  """
  def model do
    init_fn = fn _inp, init_state -> init_state end

    forward_fn = fn params, input ->
      logits = predict(params, input)
      %{prediction: logits, state: %{}}
    end

    {init_fn, forward_fn}
  end

  @doc """
  Forward for training: full-sequence logits. Params are the flat map from CheckpointLoader.
  Uses FuxiLinearInference when params are FuXi checkpoint; otherwise Inference (GPT-2).
  Input: {batch_token_ids, batch_aux_embeds, embed_mask} or
  {batch_token_ids, batch_aux_embeds, embed_mask, all_timestamps} when FuXi real timestamps.
  """
  def predict(params, input) when is_map(params) do
    {batch_token_ids, batch_aux_embeds, embed_mask, all_timestamps} =
      case input do
        {a, b, c} -> {a, b, c, nil}
        {a, b, c, t} -> {a, b, c, t}
      end

    fuxi_opts = if all_timestamps, do: [all_timestamps: all_timestamps], else: []

    if Inference.fuxi_checkpoint?(params) do
      FuxiLinearInference.forward_full_sequence(
        batch_token_ids,
        batch_aux_embeds,
        embed_mask,
        params,
        fuxi_opts
      )
    else
      Inference.forward_full_sequence(batch_token_ids, batch_aux_embeds, embed_mask, params)
    end
  end

  @doc """
  Loss function for Axon trainer: (y_true, y_pred) -> scalar (Axon passes labels first).
  Uses Training.loss_shifted_ce(logits, labels); label_ignore -100.
  """
  def loss_fn(y_true, y_pred) do
    Training.loss_shifted_ce(y_pred, y_true)
  end

  @doc """
  Builds an Axon.Loop trainer (model + loss + optimizer). Requires Axon.ModelState-compatible
  params; for flat checkpoint params use `run/3` instead.
  """
  def trainer(opts \\ []) do
    optimizer = Keyword.get(opts, :optimizer, :adam)
    log = Keyword.get(opts, :log, 50)
    Axon.Loop.trainer(model(), &loss_fn/2, optimizer, log: log)
  end

  @doc """
  Runs the training loop over a stream of batches with flat params (e.g. from
  CheckpointLoader.load_from_export/1). Uses the same forward (Inference.forward_full_sequence)
  and loss (Training.loss_shifted_ce) with Polaris optimizer.

  - `stream` - Enumerable of `{{batch_token_ids, batch_aux_embeds, embed_mask}, batch_labels}`.
  - `initial_state` - Flat map of params (from CheckpointLoader). Required for training.
  - `opts` - `:iterations` (default 1), `:log` (log every N batches), `:log_interval_sec` (log at
    least every N seconds, default 20; 0 to disable), `:optimizer` (e.g. `:adam`),
    `:learning_rate` (default 1.0e-4), `:save_every` (save checkpoint every N steps; 0 to disable),
    `:save_fn` (fn (step, params) -> :ok; called at step 0 and every save_every steps),
    `:eval_test_every` (eval test loss every N steps; 0 to disable),
    `:eval_test_fn` (fn (params) -> {:ok, loss} | {:error, _}; required when eval_test_every > 0),
    `:mtp_loss_weight` (default 0): when > 0, add MTP loss over last 4 positions (next item).

  Returns the updated flat params.
  """
  def run(stream, initial_state, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 1)
    log_every = Keyword.get(opts, :log, 50)
    log_interval_sec = Keyword.get(opts, :log_interval_sec, 20)
    check_interval = opts[:resource_check_interval] |> Kernel.||(5)
    save_every = opts[:save_every] |> Kernel.||(0)
    save_fn = Keyword.get(opts, :save_fn)
    eval_test_every = opts[:eval_test_every] |> Kernel.||(0)
    eval_test_fn = Keyword.get(opts, :eval_test_fn)

    check_opts =
      Keyword.get(opts, :resource_check_opts, [])
      |> Keyword.put_new(:start_monotonic_sec, System.monotonic_time(:second))

    {init_opt, update_opt} = optimizer_from_opts(opts)
    loss_fn_effective = build_loss_fn(Keyword.get(opts, :mtp_loss_weight, 0.0))

    batches = stream |> Enum.take(iterations)
    opt_state = init_opt.(initial_state)
    start_sec = System.monotonic_time(:second)

    # Eval and save at step 0 (before any training) when requested
    if eval_test_every > 0 and eval_test_fn do
      case eval_test_fn.(initial_state) do
        {:ok, test_loss} when is_number(test_loss) and test_loss == test_loss ->
          require Logger
          Logger.info("Step 0 (initial) test_loss=#{safe_round(test_loss, 4)}")

        _ ->
          :ok
      end
    end

    if save_every > 0 and save_fn do
      save_fn.(0, initial_state)
    end

    step_jit =
      Nx.Defn.jit(fn params, opt_state, input, labels ->
        {loss, grads} =
          Nx.Defn.value_and_grad(params, fn p ->
            logits = predict(p, input)
            loss_fn_effective.(labels, logits)
          end)

        {updates, new_opt_state} = update_opt.(grads, opt_state, params)
        new_params = Polaris.Updates.apply_updates(params, updates)
        {new_params, new_opt_state, loss}
      end)

    {final_params, _, _, _, _} =
      Enum.reduce_while(batches, {initial_state, opt_state, 0, start_sec, nil}, fn
        {input, labels}, {params, opt_state, i, last_log_sec, best_test} ->
          {new_params, new_opt_state, loss} = step_jit.(params, opt_state, input, labels)

          loss_num = Nx.to_number(loss)
          now_sec = System.monotonic_time(:second)

          # Show progress occasionally: first step, every log_every batches, or every log_interval_sec
          if log_every > 0 and rem(i, log_every) == 0 do
            require Logger
            Logger.info("Batch #{i}, loss: #{loss_num}")
          end

          last_log_sec =
            if log_interval_sec > 0 and (i == 0 or now_sec - last_log_sec >= log_interval_sec) do
              msg =
                "Step #{i}, train_loss: #{safe_round(loss_num, 6)}, elapsed #{now_sec - start_sec}s"

              padded = String.pad_trailing(msg, 80)
              IO.write(:stdio, "\r" <> padded)
              now_sec
            else
              last_log_sec
            end

          # Eval test loss every N steps
          {new_best_test, _} =
            if (eval_test_every > 0 and eval_test_fn) && rem(i + 1, eval_test_every) == 0 do
              case eval_test_fn.(new_params) do
                {:ok, test_loss} when is_number(test_loss) and not (test_loss != test_loss) ->
                  best =
                    if is_nil(best_test) or test_loss < best_test, do: test_loss, else: best_test

                  require Logger

                  Logger.info(
                    "Step #{i + 1} train_loss=#{safe_round(loss_num, 4)} test_loss=#{safe_round(test_loss, 4)} best_test=#{safe_round(best || test_loss, 4)}"
                  )

                  {best, :ok}

                {:ok, _} ->
                  require Logger
                  Logger.warning("Step #{i + 1} test_loss=nan or invalid")
                  {best_test, :ok}

                {:error, reason} ->
                  require Logger
                  Logger.warning("Step #{i + 1} test loss failed: #{inspect(reason)}")
                  {best_test, :ok}
              end
            else
              {best_test, :ok}
            end

          if (is_integer(save_every) and save_every > 0 and save_fn) &&
               rem(i + 1, save_every) == 0 do
            save_fn.(i + 1, new_params)
          end

          if is_integer(check_interval) and check_interval > 0 and rem(i + 1, check_interval) == 0 do
            case RecGPT.ResourceCheck.check(check_opts) do
              :ok ->
                {:cont, {new_params, new_opt_state, i + 1, last_log_sec, new_best_test}}

              {:halt, reason} ->
                require Logger
                Logger.warning("Pretrain circuit break: #{reason}")
                {:halt, {new_params, new_opt_state, i + 1, last_log_sec, new_best_test}}
            end
          else
            {:cont, {new_params, new_opt_state, i + 1, last_log_sec, new_best_test}}
          end
      end)

    final_params
  end

  defp build_loss_fn(mtp_weight) when is_number(mtp_weight) and mtp_weight > 0 do
    fn labels, logits ->
      l_ce = Training.loss_shifted_ce(logits, labels)
      l_mtp = Training.loss_mtp_last_4(logits, labels)
      Nx.add(l_ce, Nx.multiply(mtp_weight, l_mtp))
    end
  end

  defp build_loss_fn(_), do: &Training.loss_shifted_ce(&2, &1)

  defp safe_round(x, precision) when is_number(x) and x == x, do: Float.round(x, precision)
  defp safe_round(_, _), do: "nan"

  defp optimizer_from_opts(opts) do
    lr = Keyword.get(opts, :learning_rate, 1.0e-4)

    case Keyword.get(opts, :optimizer, :adam) do
      :adam ->
        Polaris.Optimizers.adam(learning_rate: lr)

      :sgd ->
        Polaris.Optimizers.sgd(learning_rate: Keyword.get(opts, :learning_rate, 1.0e-3))

      other when is_tuple(other) ->
        other
    end
  end

  @doc """
  Builds a stream of batches for training from sequences, token_id_list, and item_embeddings.

  - `seqs` - list of sequences (list of item indices).
  - `token_id_list` - list of 4-token lists per item (from FSQ encoder).
  - `item_embeddings` - tensor (num_items, 768).
  - `opts` - `batch_size` (default 8), `:epochs` (default 1), `:shuffle` (default true).

  Yields `{{batch_token_ids, batch_aux_embeds, embed_mask}, batch_labels}` as required by
  the trainer.
  """
  def stream_batches(seqs, token_id_list, item_embeddings, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 8)
    epochs = Keyword.get(opts, :epochs, 1)
    shuffle = Keyword.get(opts, :shuffle, true)
    timestamps = Keyword.get(opts, :timestamps)

    indices = 0..(length(seqs) - 1)//1 |> Enum.to_list()

    Stream.flat_map(1..epochs, fn _epoch ->
      indices = if shuffle, do: Enum.shuffle(indices), else: indices

      indices
      |> Enum.chunk_every(batch_size)
      |> Stream.map(fn batch_indices ->
        {batch_seq, batch_labels, batch_aux_embeds, embed_mask, all_timestamps} =
          Training.build_train_batch(
            seqs,
            token_id_list,
            item_embeddings,
            batch_indices,
            timestamps
          )

        inputs =
          if all_timestamps do
            {batch_seq, batch_aux_embeds, embed_mask, all_timestamps}
          else
            {batch_seq, batch_aux_embeds, embed_mask}
          end

        {inputs, batch_labels}
      end)
    end)
  end

  @doc """
  Stream of batches for training from DB: constant memory over the full dataset and multiple epochs.

  Loads only seq_ids up front (small); for each batch loads that batch's sequence rows from the DB.
  Use when `train_path` is `:db` to train for 5 (or any) epochs without loading all sequences into RAM.

  - `seq_ids` - list of train sequence IDs (e.g. from `RecGPT.Catalog.Scan.load_train_seq_ids/1`).
  - `token_id_list` - list of 4-token lists per item (from fixture or DB).
  - `item_embeddings` - tensor (num_items, 768).
  - `opts` - `batch_size` (default 8), `:epochs` (default 1), `:shuffle` (default true), `:repo`.

  Yields the same format as `stream_batches/4`.
  """
  def stream_batches_from_db(seq_ids, token_id_list, item_embeddings, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 8)
    epochs = Keyword.get(opts, :epochs, 1)
    shuffle = Keyword.get(opts, :shuffle, true)
    scan_opts = Keyword.take(opts, [:repo])

    Stream.flat_map(1..epochs, fn _epoch ->
      chunk_ids =
        if(shuffle, do: Enum.shuffle(seq_ids), else: seq_ids)
        |> Enum.chunk_every(batch_size)

      Stream.map(chunk_ids, fn batch_seq_ids ->
        {sequences, timestamps} =
          RecGPT.Catalog.Scan.load_sequences_by_seq_ids(batch_seq_ids, scan_opts)

        batch_indices = 0..(length(sequences) - 1)//1 |> Enum.to_list()

        {batch_seq, batch_labels, batch_aux_embeds, embed_mask, all_timestamps} =
          Training.build_train_batch(
            sequences,
            token_id_list,
            item_embeddings,
            batch_indices,
            timestamps
          )

        inputs =
          if all_timestamps do
            {batch_seq, batch_aux_embeds, embed_mask, all_timestamps}
          else
            {batch_seq, batch_aux_embeds, embed_mask}
          end

        {inputs, batch_labels}
      end)
    end)
  end
end
