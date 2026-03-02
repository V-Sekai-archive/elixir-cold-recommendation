defmodule RecGPT.StaffApi do
  @moduledoc """
  Staff/admin API service for managing catalogues, sequences, fixture build, and pretraining.

  Use this module when building a staff-facing API (e.g. gRPC or HTTP) to:
  - List, get, create, update catalogue items
  - Sync items or sequences from JSON files
  - Build fixture and write to path
  - Run pretraining
  - Manage canonical item texts (for RecGPT parity)

  All functions return `{:ok, result}` or `{:error, reason}`. The default implementation
  uses RecGPT.Catalog.Sync, RecGPT.FixtureBuild, RecGPT.Repo, and RecGPT.PretrainRunner.

  **SPMD compatibility:** Operations are scoped by explicit parameters (path, data_dir, etc.);
  there is no implicit global state in the contract. The gRPC layer adds optional `rank` to
  requests for multi-rank deployments. Single-rank implementations ignore rank.
  """
  @callback list_items(source :: :db | {:path, String.t()}) ::
              {:ok, [map()]} | {:error, term()}
  @callback get_item(item_id :: non_neg_integer()) :: {:ok, map() | nil} | {:error, term()}
  @callback upsert_items(entries :: [%{required(:item_id) => integer(), required(:title) => String.t()}]) ::
              :ok | {:error, term()}
  @callback sync_items_from_json(path :: String.t()) :: :ok | {:error, term()}
  @callback write_items_json(path :: String.t(), items :: [map()]) :: :ok | {:error, term()}
  @callback sync_sequences(data_dir :: String.t()) :: :ok | {:error, term()}
  @callback build_fixture(items_path :: String.t(), ckpt_dir :: String.t(), opts :: keyword()) ::
              {:ok, map()} | {:error, term()}
  @callback write_fixture(fixture :: map(), path :: String.t()) :: :ok | {:error, term()}
  @callback pretrain(opts :: keyword()) :: :ok | {:error, term()}
  @callback set_canonical_texts(entries :: [%{required(:item_id) => integer(), required(:text) => binary()}]) ::
              :ok | {:error, term()}

  @doc "Returns the configured implementation (default: RecGPT.StaffApi.Default)."
  def impl do
    Application.get_env(:recgpt, :staff_api_impl, RecGPT.StaffApi.Default)
  end

  def list_items(source \\ :db), do: impl().list_items(source)
  def get_item(item_id), do: impl().get_item(item_id)
  def upsert_items(entries), do: impl().upsert_items(entries)
  def sync_items_from_json(path), do: impl().sync_items_from_json(path)
  def write_items_json(path, items), do: impl().write_items_json(path, items)
  def sync_sequences(data_dir), do: impl().sync_sequences(data_dir)
  def build_fixture(items_path, ckpt_dir, opts \\ []), do: impl().build_fixture(items_path, ckpt_dir, opts)
  def write_fixture(fixture, path), do: impl().write_fixture(fixture, path)
  def pretrain(opts), do: impl().pretrain(opts)
  def set_canonical_texts(entries), do: impl().set_canonical_texts(entries)
end
