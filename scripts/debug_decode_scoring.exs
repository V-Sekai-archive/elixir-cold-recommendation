# Isolated debug for decode scoring and trie.
# Run: mix run scripts/debug_decode_scoring.exs
#
# Finding: if the fixture's token_id_list has the same first token for every item,
# the trie has one path -> valid_next_tokens(trie, []) returns 1 -> beam stays 1 -> single item.

Application.ensure_all_started(:nx)
# Use BinaryBackend so we don't need EXLA for the scoring test
Nx.default_backend(Nx.BinaryBackend)

IO.puts("=== 1. Scoring pipeline (stub logits, stub entries) ===")
batch_size = 1
vocab_size = 5
logits = Nx.tensor([[10.0, 20.0, 5.0, 1.0, 0.0]])
entries = [
  {0, 0, [], 0.0},
  {0, 1, [], 0.0},
  {0, 2, [], 0.0},
]

flat_logits = Nx.reshape(logits, {batch_size * vocab_size})
scores_list =
  Enum.map(entries, fn {b, t, _, _} ->
    idx = b * vocab_size + t
    flat_logits |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number()
  end)

IO.inspect(scores_list, label: "scores_list")
all_candidates =
  Enum.zip(entries, scores_list)
  |> Enum.map(fn {{_b, token_id, prefix, parent_score}, logit} ->
    {prefix ++ [token_id], parent_score + logit}
  end)

result_beam = all_candidates |> Enum.sort_by(fn {_, s} -> s end, :desc) |> Enum.take(2)
IO.inspect(result_beam, label: "result_beam")

if length(scores_list) == 3 and length(result_beam) == 2 do
  IO.puts("  -> OK: 3 scores, 2 candidates\n")
else
  IO.puts("  -> FAIL: expected 3 scores and 2 candidates\n")
end

IO.puts("=== 2. Real trie from fixture (valid_next_tokens for prefix []) ===")
fixture_path = Path.expand("data/steam/fixture.json", File.cwd!())

if File.regular?(fixture_path) do
  fixture = File.read!(fixture_path) |> Jason.decode!()
  token_id_list =
    (fixture["token_id_list"] || [])
    |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

  trie = RecGPT.Trie.build(token_id_list)
  first_tokens = RecGPT.Trie.valid_next_tokens(trie, [])
  num_first = length(first_tokens)
  IO.puts("  num valid first tokens (prefix []): #{num_first}")
  IO.inspect(Enum.take(first_tokens, 10), label: "  first 10 token IDs")

  if num_first == 1 do
    IO.puts("  -> TRIE HAS ONE PATH: beam will always have 1 candidate -> single item result\n")
  else
    IO.puts("  -> trie has #{num_first} first tokens (beam can expand)\n")
  end
else
  IO.puts("  fixture not found: #{fixture_path}\n")
end

IO.puts("=== 3. One full step with real trie + stub logits ===")
if File.regular?(fixture_path) do
  fixture = File.read!(fixture_path) |> Jason.decode!()
  token_id_list =
    (fixture["token_id_list"] || [])
    |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

  trie = RecGPT.Trie.build(token_id_list)
  beam = [{[], 0.0}]
  # Stub batch_fn: return logits {1, vocab_size}
  vocab_size = 15_361
  stub_logits = Nx.broadcast(0.0, {1, vocab_size}) |> Nx.as_type({:f, 32})
  # Make token 100 slightly higher so we can see ordering
  stub_logits = Nx.put_slice(stub_logits, [0, 100], Nx.tensor([[1.0]]))
  batch_fn = fn _prefixes, _cache -> {stub_logits, nil} end

  full_prefixes = Enum.map(beam, fn {prefix, _} -> [] ++ prefix end)
  {logits, _cache} = batch_fn.(full_prefixes, nil)
  {batch_size, vocab_size} = Nx.shape(logits)

  entries =
    Enum.flat_map(0..(batch_size - 1), fn i ->
      {prefix, parent_score} = Enum.at(beam, i)
      valid = RecGPT.Trie.valid_next_tokens(trie, prefix)
      Enum.map(valid, fn token_id -> {i, token_id, prefix, parent_score} end)
    end)

  IO.puts("  entries count: #{length(entries)}")

  if entries != [] do
    flat_logits = Nx.reshape(logits, {batch_size * vocab_size})
    scores_list =
      Enum.map(entries, fn {b, t, _, _} ->
        idx = b * vocab_size + t
        flat_logits |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number()
      end)

    all_candidates =
      Enum.zip(entries, scores_list)
      |> Enum.map(fn {{_b, token_id, prefix, parent_score}, logit} ->
        {prefix ++ [token_id], parent_score + logit}
      end)

    result_beam = all_candidates |> Enum.sort_by(fn {_, s} -> s end, :desc) |> Enum.take(10)
    IO.puts("  result_beam count: #{length(result_beam)}")
    IO.inspect(Enum.take(result_beam, 3), label: "  first 3 candidates")
  end
else
  IO.puts("  (skip: no fixture)")
end
