# Benchmark Serve.recommend/3. Run: mix run bench/recgpt_serve_bench.exs

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
state = %RecGPT.Serve{params: params, trie: trie, token_id_list: token_id_list, item_text: %{}, num_items: 2, get_logits_fn: get_logits_fn}
Benchee.run(%{"recommend/3" => fn -> RecGPT.Serve.recommend(state, [0, 1], 5) end}, time: 3, warmup: 1)
