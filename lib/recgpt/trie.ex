defmodule RecGPT.Trie do
  @moduledoc """
  Catalog trie: maps 4-token FSQ sequences to item_id for RecGPT decode.

  Built from `token_id_list` (list of 4-token lists, one per catalog item).
  Supports lookup of full 4-token sequence → item_id and valid next tokens
  at each prefix for beam search.
  """

  @seq_len 4

  @doc """
  Build a trie from token_id_list. Each element is a list of 4 token IDs (0..vocab_size-1).
  Returns an opaque trie (nested map); use `lookup/2` and `valid_next_tokens/2`.
  """
  def build(token_id_list) when is_list(token_id_list) do
    Enum.reduce(Enum.with_index(token_id_list), %{}, fn {tokens, item_id}, acc ->
      case tokens do
        [t0, t1, t2, t3] when is_integer(t0) and is_integer(t1) and is_integer(t2) and is_integer(t3) ->
          put_path(acc, [t0, t1, t2, t3], item_id)
        _ ->
          acc
      end
    end)
  end

  defp put_path(map, [k], v), do: Map.put(map, k, v)
  defp put_path(map, [k | rest], v) do
    child = Map.get(map, k) || %{}
    Map.put(map, k, put_path(child, rest, v))
  end

  @doc """
  Lookup item_id for a complete 4-token sequence. Returns `{:ok, item_id}` or `:not_found`.
  """
  def lookup(trie, [t0, t1, t2, t3]) when is_map(trie) do
    case get_in(trie, [t0, t1, t2, t3]) do
      nil -> :not_found
      item_id when is_integer(item_id) -> {:ok, item_id}
      _ -> :not_found
    end
  end

  def lookup(_trie, _token_list), do: :not_found

  @doc """
  Return list of valid next token IDs that extend `prefix` to some catalog item.
  `prefix` is 0..3 tokens (e.g. [] for first token, [t0] for second, [t0,t1,t2] for fourth).
  """
  def valid_next_tokens(trie, []) when is_map(trie), do: Map.keys(trie)
  def valid_next_tokens(trie, [h | t]) when is_map(trie) do
    case Map.get(trie, h) do
      nil -> []
      next when is_map(next) -> valid_next_tokens(next, t)
      _ -> []
    end
  end
  def valid_next_tokens(_, prefix) when not is_list(prefix), do: []

  def valid_next_tokens(_, _), do: []

  @doc "Number of tokens per item (4)."
  def seq_len, do: @seq_len
end
