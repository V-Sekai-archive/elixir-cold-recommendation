# Benchmark Serve.recommend/3. Run in dev only: MIX_ENV=dev mix run bench/recgpt_serve_bench.exs
#
# NOTE: This uses a minimal stub state (no trie_tensors, get_logits_batch_tensor_fn, etc.).
# Serve.recommend/3 returns {:error, _} immediately, so results measure the error path only.
# For real recommendation latency, use: mix recgpt.trace_predict --runs 20
# or run mix recgpt.serve and call the gRPC Predict API repeatedly (see docs/08_latency_and_performance.md).

unless Mix.env() == :dev do
  raise "Benchee benchmark is dev-only. Run: MIX_ENV=dev mix run bench/recgpt_serve_bench.exs"
end

Application.ensure_all_started(:nx)
token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
trie = RecGPT.Trie.build(token_id_list)

params = %{
  "wte" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
  "pred_head.weight" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
  "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
}

get_logits_fn = fn token_list ->
  seq_len = length(token_list)
  batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
  batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
  embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
  RecGPT.Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
end

state = %RecGPT.Serve{
  params: params,
  trie: trie,
  token_id_list: token_id_list,
  item_text: %{},
  num_items: 2,
  get_logits_fn: get_logits_fn
}

Benchee.run(%{"recommend/3" => fn -> RecGPT.Serve.recommend(state, [0, 1], 5) end},
  time: 3,
  warmup: 1
)
