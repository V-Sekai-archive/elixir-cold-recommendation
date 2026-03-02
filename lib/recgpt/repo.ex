defmodule RecGPT.Repo do
  @moduledoc """
  Ecto repo for SQLite catalog and token storage.
  Used when building fixture to SQLite or loading state from SQLite.
  """
  use Ecto.Repo,
    otp_app: :recgpt,
    adapter: Ecto.Adapters.SQLite3
end
