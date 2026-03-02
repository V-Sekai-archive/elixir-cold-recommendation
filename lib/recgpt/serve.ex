defmodule RecGPT.Serve do
  @moduledoc """
  Next-item recommendation (backend for gRPC API).

  Implements `RecGPT.RecommendationService`; used as the default implementation
  when the application calls the recommendation service. Loads model + token_id_list + trie once.
  Served via gRPC (recgpt.v1.PredictionService). Run: mix recgpt.serve [--grpc-port 50051].
  Requires fixture and checkpoint export dir.
  """
  @behaviour RecGPT.RecommendationService

  alias RecGPT.CheckpointLoader
  alias RecGPT.Decode
  alias RecGPT.Inference
  alias RecGPT.Trie

  @padding_id 15_360
  @max_length 255
  @seq_token_capacity 1024

  defstruct [:params, :trie, :token_id_list, :token_id_map, :item_text, :num_items, :get_logits_fn]

  @type state :: %__MODULE__{
          params: map(),
          trie: map(),
          token_id_list: [[non_neg_integer()]],
          token_id_map: %{non_neg_integer() => [non_neg_integer()]} | nil,
          item_text: %{non_neg_integer() => String.t() | map()},
          num_items: non_neg_integer(),
          get_logits_fn: (list(non_neg_integer()) -> Nx.Tensor.t())
        }

  @doc """
  Load server state: checkpoint export, fixture (token_id_list), optional catalog JSON.
  Returns {:ok, state} or {:error, reason}.
  """
  @spec load_state(String.t(), String.t(), String.t() | nil) ::
          {:ok, state()} | {:error, String.t()}
  def load_state(fixture_path, ckpt_export_dir, catalog_path \\ nil) do
    with {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {:ok, token_id_list, num_items} <- load_fixture(fixture_path),
         {:ok, item_text} <- load_catalog(catalog_path, num_items) do
      trie = Trie.build(token_id_list)
      get_logits_fn = build_get_logits_fn(params)

      state = %__MODULE__{
        params: params,
        trie: trie,
        token_id_list: token_id_list,
        token_id_map: nil,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn
      }

      {:ok, state}
    end
  end

  @doc """
  Load state from catalog DB (item_tokens) + checkpoint. Constant memory: streams item_tokens into trie and map.
  Requires RECGPT_SQLITE_PATH and mix ecto.migrate. Run build_fixture with --sqlite (or RECGPT_SQLITE_PATH) first.
  """
  @spec load_state_from_db(String.t(), String.t() | nil) :: {:ok, state()} | {:error, String.t()}
  def load_state_from_db(ckpt_export_dir, catalog_path \\ nil) do
    with {:ok, params} <- load_checkpoint(ckpt_export_dir),
         {:ok, trie, token_id_map, num_items} <- load_fixture_from_db(),
         {:ok, item_text} <- load_catalog(catalog_path, num_items) do
      get_logits_fn = build_get_logits_fn(params)
      state = %__MODULE__{
        params: params,
        trie: trie,
        token_id_list: [],
        token_id_map: token_id_map,
        item_text: item_text,
        num_items: num_items,
        get_logits_fn: get_logits_fn
      }
      {:ok, state}
    end
  end

  defp load_fixture_from_db do
    import Ecto.Query
    alias RecGPT.Catalog.ItemToken
    alias RecGPT.Repo

    stream =
      from(t in ItemToken, order_by: [asc: t.item_id], select: {t.item_id, t.t0, t.t1, t.t2, t.t3})
      |> Repo.stream()

    stream =
      Stream.map(stream, fn {item_id, t0, t1, t2, t3} ->
        tokens = [t0 || 0, t1 || 0, t2 || 0, t3 || 0]
        {item_id, tokens}
      end)

    {trie, token_id_map, num_items} =
      Enum.reduce(stream, {%{}, %{}, 0}, fn {item_id, tokens}, {acc_trie, acc_map, max_id} ->
        new_trie = Trie.add_item(acc_trie, item_id, tokens)
        new_map = Map.put(acc_map, item_id, tokens)
        new_max = max(item_id + 1, max_id)
        {new_trie, new_map, new_max}
      end)

    {:ok, trie, token_id_map, num_items}
  end

  defp load_checkpoint(dir) do
    manifest = Path.join(dir, "manifest.json")

    if File.regular?(manifest) do
      {:ok, CheckpointLoader.load_from_export(dir)}
    else
      {:error, "checkpoint not found: #{dir}"}
    end
  end

  defp load_fixture(path) do
    if File.regular?(path) do
      fixture = File.read!(path) |> Jason.decode!()

      token_id_list =
        (fixture["token_id_list"] || []) |> Enum.map(&Enum.map(&1, fn x -> round(x) end))

      num_items = fixture["num_items"] || length(token_id_list)
      {:ok, token_id_list, num_items}
    else
      {:error, "fixture not found: #{path}"}
    end
  end

  defp load_catalog(nil, _num_items), do: {:ok, %{}}
  defp load_catalog(path, _num_items) when path in [nil, "", []], do: {:ok, %{}}

  defp load_catalog(path, _num_items) do
    if File.regular?(path) do
      raw = File.read!(path) |> Jason.decode!()

      item_text =
        case raw do
          %{"items" => items} when is_list(items) ->
            Enum.reduce(items, %{}, fn item, acc ->
              id = item["id"] || item["item_id"]
              text = item["text"] || item["title"] || item["raw"] || ""
              if is_integer(id), do: Map.put(acc, id, text), else: acc
            end)

          _ ->
            %{}
        end

      {:ok, item_text}
    else
      {:ok, %{}}
    end
  end

  defp build_get_logits_fn(params) do
    fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end
  end

  @doc """
  Convert item_ids (catalog indices) to left-padded token sequence for inference (same as Python serve seq_to_batch).
  Uses state.token_id_list when present, else state.token_id_map (when loaded from DB).
  """
  @spec item_ids_to_context_token_ids(
          [non_neg_integer()],
          [[non_neg_integer()]] | state(),
          non_neg_integer() | nil
        ) :: [integer()]
  def item_ids_to_context_token_ids(item_ids, token_id_list, padding_id \\ @padding_id)
      when is_list(token_id_list) or is_struct(token_id_list) do
    seq = Enum.take(item_ids, -@max_length)
    token_list =
      if is_struct(token_id_list) do
        state = token_id_list
        Enum.flat_map(seq, fn iid ->
          if state.token_id_map && state.token_id_map != %{} do
            Map.get(state.token_id_map, iid, [0, 0, 0, 0])
          else
            Enum.at(state.token_id_list || [], iid) || [0, 0, 0, 0]
          end
        end)
      else
        Enum.flat_map(seq, fn iid -> Enum.at(token_id_list, iid) || [0, 0, 0, 0] end)
      end
    len = length(token_list)
    padding = List.duplicate(padding_id || @padding_id, @seq_token_capacity - len)
    padding ++ token_list
  end

  @doc """
  Recommend next item(s) given context item_ids. Returns up to `top_k` item_ids (best first) from beam search.
  """
  @spec recommend(state(), [non_neg_integer()], pos_integer()) ::
          {:ok, [non_neg_integer()]} | {:error, String.t()}
  def recommend(state, item_ids, top_k \\ 5)
      when is_list(item_ids) and is_integer(top_k) and top_k >= 1 do
    if item_ids == [] do
      {:error, "item_ids must be non-empty"}
    else
      top_k = min(top_k, 20)
      context_token_ids = item_ids_to_context_token_ids(item_ids, state)

      case Decode.beam_search_top_k(state.get_logits_fn, state.trie, context_token_ids, top_k) do
        {:ok, list} -> {:ok, list}
        :not_found -> {:ok, []}
      end
    end
  end
end
