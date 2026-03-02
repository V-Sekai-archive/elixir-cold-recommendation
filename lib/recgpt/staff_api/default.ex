defmodule RecGPT.StaffApi.Default do
  @moduledoc """
  Default implementation of RecGPT.StaffApi. Uses RecGPT.Catalog.Sync,
  RecGPT.FixtureBuild, RecGPT.Repo, and RecGPT.PretrainRunner.
  """
  @behaviour RecGPT.StaffApi

  import Ecto.Query
  alias RecGPT.Catalog
  alias RecGPT.Catalog.CanonicalItemText
  alias RecGPT.Catalog.Item
  alias RecGPT.Catalog.Sync
  alias RecGPT.FixtureBuild
  alias RecGPT.PretrainRunner
  alias RecGPT.Repo

  @impl true
  def list_items(:db) do
    Application.ensure_all_started(:recgpt)
    items =
      from(i in Item, order_by: [asc: i.item_id])
      |> Repo.all()

    result = Enum.map(items, &%{item_id: &1.item_id, title: &1.title})
    {:ok, result}
  rescue
    e -> {:error, e}
  end

  def list_items({:path, path}) when is_binary(path) do
    raw = File.read!(path) |> Jason.decode!()
    items = raw["items"] || []
    {:ok, items}
  rescue
    e -> {:error, e}
  end

  @impl true
  def get_item(item_id) when is_integer(item_id) do
    Application.ensure_all_started(:recgpt)
    case Repo.get(Item, item_id) do
      nil -> {:ok, nil}
      %Item{} = item -> {:ok, %{item_id: item.item_id, title: item.title}}
    end
  rescue
    e -> {:error, e}
  end

  @impl true
  def upsert_items(entries) when is_list(entries) do
    Application.ensure_all_started(:recgpt)
    maps = Enum.map(entries, fn %{item_id: id, title: title} -> %{item_id: id, title: title} end)
    Repo.insert_all(Item, maps, on_conflict: {:replace, [:title]}, conflict_target: [:item_id])
    :ok
  rescue
    e -> {:error, e}
  end

  @impl true
  def sync_items_from_json(path) when is_binary(path) do
    Application.ensure_all_started(:recgpt)
    raw = File.read!(path) |> Jason.decode!()
    items = raw["items"] || []
    entries = Enum.map(items, fn item -> %{item_id: item["id"] || item["item_id"], title: item["title"] || item["text"] || ""} end)
    Sync.clear_catalog_tables()
    Enum.chunk_every(entries, 1000) |> Enum.each(&Sync.insert_items/1)
    :ok
  rescue
    e -> {:error, e}
  end

  @impl true
  def write_items_json(path, items) when is_list(items) do
    content = %{"items" => items, "num_items" => length(items)}
    Catalog.write!(path, content)
    :ok
  rescue
    e -> {:error, e}
  end

  @impl true
  def sync_sequences(data_dir) when is_binary(data_dir) do
    Application.ensure_all_started(:recgpt)
    train_path = Path.join(data_dir, "train_sequences.json")
    cold_train_path = Path.join(data_dir, "cold_train_sequences.json")
    test_path = Path.join(data_dir, "test_sequences.json")
    cold_test_path = Path.join(data_dir, "cold_test_sequences.json")
    Sync.sync_sequences_from_json(train_path, cold_train_path)
    Sync.sync_test_from_json(test_path, cold_test_path)
    :ok
  rescue
    e -> {:error, e}
  end

  @impl true
  def build_fixture(items_path, ckpt_dir, opts \\ []) do
    Application.ensure_all_started(:recgpt)
    fixture = FixtureBuild.build(items_path, ckpt_dir, opts)
    {:ok, fixture}
  rescue
    e -> {:error, e}
  end

  @impl true
  def write_fixture(fixture, path) when is_map(fixture) and is_binary(path) do
    FixtureBuild.write_fixture(fixture, path)
    :ok
  rescue
    e -> {:error, e}
  end

  @impl true
  def pretrain(opts) when is_list(opts) do
    runner_opts =
      opts
      |> Enum.map(fn
        {:ckpt_dir, v} -> {:ckpt_dir, v}
        {:ckpt, v} -> {:ckpt_dir, v}
        {:fixture_path, v} -> {:fixture_path, v}
        {:fixture, v} -> {:fixture_path, v}
        {:train_path, v} -> {:train_path, v}
        {:train, v} -> {:train_path, v}
        {:items_path, v} -> {:items_path, v}
        {:items, v} -> {:items_path, v}
        {:out_dir, v} -> {:out_dir, v}
        {:out, v} -> {:out_dir, v}
        {:limit, v} -> {:limit, v}
        {:iterations, v} -> {:iterations, v}
        {:batch_size, v} -> {:batch_size, v}
        {:learning_rate, v} -> {:learning_rate, v}
        {:log, v} -> {:log, v}
        {:log_interval_sec, v} -> {:log_interval_sec, v}
        {:resource_check_opts, v} -> {:resource_check_opts, v}
        _ -> nil
      end)
      |> Enum.reject(&is_nil/1)

    PretrainRunner.run(runner_opts)
  end

  @impl true
  def set_canonical_texts(entries) when is_list(entries) do
    Application.ensure_all_started(:recgpt)
    Repo.delete_all(CanonicalItemText)
    maps = Enum.map(entries, fn %{item_id: id, text: text} -> %{item_id: id, text: text} end)
    Enum.chunk_every(maps, 1000) |> Enum.each(&Repo.insert_all(CanonicalItemText, &1))
    :ok
  rescue
    e -> {:error, e}
  end
end
