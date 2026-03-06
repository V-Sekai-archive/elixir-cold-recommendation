defmodule RecGPT.VisionContrastive do
  @moduledoc """
  Contrastive training for vision–text alignment: projector only (DINOv2 and MPNet frozen).

  Uses InfoNCE loss: projected vision 768-d vs text 768-d; positive pair = same index in batch.
  Batches are {vision_768, text_768} with shapes [batch_size, 768]. Both encoders are assumed
  frozen; only VisionProjector params are updated.
  """

  alias RecGPT.VisionProjector
  alias RecGPT.Training

  @default_temperature 0.07

  @doc """
  InfoNCE contrastive loss for a batch of (vision_768, text_768) pairs.

  - `proj_params` — VisionProjector params (map).
  - `vision_768` — [batch_size, 768] from DINOv2 (frozen).
  - `text_768` — [batch_size, 768] from MPNet (frozen), L2-normalized.
  - `opts` — `:temperature` (default 0.07).

  Returns scalar loss. Positive pair: (vision_768[i], text_768[i]).
  """
  def loss(proj_params, vision_768, text_768, opts \\ []) do
    temp = Keyword.get(opts, :temperature, @default_temperature)
    proj_v = VisionProjector.forward(proj_params, vision_768)
    # proj_v and text_768 are L2-normed -> dot product = cosine similarity
    logits = Nx.dot(proj_v, [1], text_768, [1])
    logits = Nx.divide(logits, Nx.tensor(temp, type: Nx.type(logits)))
    batch_size = Nx.axis_size(vision_768, 0)
    labels = Nx.iota({batch_size}, type: {:s, 64})
    # Reshape for Training.loss_shifted_ce: (batch, seq_len=1, vocab=batch), labels (batch, 1)
    logits_3d = Nx.reshape(logits, {batch_size, 1, batch_size})
    labels_2d = Nx.reshape(labels, {batch_size, 1})
    Training.loss_shifted_ce(logits_3d, labels_2d)
  end

  @doc """
  Runs contrastive training for `steps` batches. Uses synthetic data if no stream given.
  Logs loss every `:log_every` steps.

  - `proj_params` — initial VisionProjector params.
  - `opts` — `:steps` (default 500), `:batch_size` (default 32), `:learning_rate` (1.0e-4),
    `:log_every` (50), `:stream` (optional enumerable of {vision_768, text_768} tensors).
  Returns updated projector params.
  """
  def run(proj_params, opts \\ []) do
    steps = Keyword.get(opts, :steps, 500)
    batch_size = Keyword.get(opts, :batch_size, 32)
    lr = Keyword.get(opts, :learning_rate, 1.0e-4)
    log_every = Keyword.get(opts, :log_every, 50)
    stream = Keyword.get(opts, :stream)

    {init_opt, update_opt} = Polaris.Optimizers.adam(learning_rate: lr)
    opt_state = init_opt.(proj_params)

    batches =
      if stream do
        Stream.take(stream, steps)
      else
        Stream.repeatedly(fn -> synthetic_batch(batch_size) end) |> Stream.take(steps)
      end

    step_fn =
      Nx.Defn.jit(fn params, opt_state, {vision_768, text_768} ->
        {loss_val, grads} =
          Nx.Defn.value_and_grad(params, fn p ->
            RecGPT.VisionContrastive.loss(p, vision_768, text_768, temperature: @default_temperature)
          end)

        {updates, new_opt_state} = update_opt.(grads, opt_state, params)
        new_params = Polaris.Updates.apply_updates(params, updates)
        {new_params, new_opt_state, loss_val}
      end)

    batches
    |> Enum.with_index(1)
    |> Enum.reduce_while({proj_params, opt_state}, fn {batch, i}, {params, opt_state} ->
      {new_params, new_opt_state, loss_val} = step_fn.(params, opt_state, batch)
      loss_num = Nx.to_number(loss_val)
      if rem(i, log_every) == 0 or i == 1, do: IO.puts("  step #{i} loss=#{Float.round(loss_num, 4)}")
      {:cont, {new_params, new_opt_state}}
    end)
    |> then(fn {final_params, _} -> final_params end)
  end

  @doc """
  Builds a stream of {vision_768, text_768} batches from precomputed .npy files.

  Expects `dataset_dir` to contain `vision_768.npy` and `text_768.npy` with shape {N, 768}.
  Options: `:batch_size` (default 32), `:shuffle` (default true), `:epochs` (default 1).
  Yields batches as `{vision_batch, text_batch}` Nx tensors (f32).
  """
  def stream_from_dataset_dir(dataset_dir, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)
    epochs = Keyword.get(opts, :epochs, 1)

    vision_path = Path.join(dataset_dir, "vision_768.npy")
    text_path = Path.join(dataset_dir, "text_768.npy")
    unless File.regular?(vision_path) and File.regular?(text_path) do
      raise "Missing vision_768.npy or text_768.npy in #{dataset_dir}"
    end

    {:ok, vision_nx} = Npy.load(vision_path, :nx)
    {:ok, text_nx} = Npy.load(text_path, :nx)
    vision = ensure_f32_2d(vision_nx)
    text = ensure_f32_2d(text_nx)
    n = elem(Nx.shape(vision), 0)
    unless elem(Nx.shape(text), 0) == n do
      raise "vision and text .npy have different row counts"
    end

    Stream.flat_map(1..epochs, fn _ ->
      indices = if shuffle, do: Enum.shuffle(0..(n - 1)), else: Enum.to_list(0..(n - 1))
      indices
      |> Enum.chunk_every(batch_size)
      |> Stream.map(fn chunk ->
        idx = Nx.tensor(chunk, type: {:s, 64}) |> Nx.new_axis(-1)
        vision_batch = Nx.gather(vision, idx) |> Nx.squeeze(axes: [1])
        text_batch = Nx.gather(text, idx) |> Nx.squeeze(axes: [1])
        {vision_batch, text_batch}
      end)
    end)
  end

  defp ensure_f32_2d(tensor) do
    tensor = Nx.as_type(tensor, {:f, 32})
    case Nx.shape(tensor) do
      {_n, 768} -> tensor
      {_n, 1, 768} -> Nx.squeeze(tensor, axes: [1])
      _ -> tensor
    end
  end

  defp synthetic_batch(batch_size) do
    # Random 768-d (standard normal * 0.02); positive = same index
    key = Nx.Random.key(:erlang.unique_integer([:positive]))
    {vision_768, key} = Nx.Random.normal(key, shape: {batch_size, 768}, type: {:f, 32})
    {text_768, _} = Nx.Random.normal(key, shape: {batch_size, 768}, type: {:f, 32})
    vision_768 = Nx.multiply(vision_768, 0.02)
    text_768 = Nx.multiply(text_768, 0.02) |> l2_norm()
    {vision_768, text_768}
  end

  defp l2_norm(x) do
    norm = Nx.LinAlg.norm(x, axes: [-1], keep_axes: true)
    norm = Nx.max(norm, Nx.tensor(1.0e-8, type: Nx.type(x)))
    Nx.divide(x, norm)
  end
end
