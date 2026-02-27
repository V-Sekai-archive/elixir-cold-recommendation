defmodule RecGPT.Repo do
  @moduledoc """
  Ecto repo for RecGPT SQLite data (e.g. UCI Clickstream).

  Configure in config/config.exs or via RECGPT_DATABASE_PATH.
  Default: data/clickstream/recgpt.db
  """
  use Ecto.Repo,
    otp_app: :recgpt,
    adapter: Ecto.Adapters.SQLite3

  def init(_type, config) do
    path =
      System.get_env("RECGPT_DATABASE_PATH") ||
        Keyword.get(config, :database) ||
        Path.join(File.cwd!(), "data/clickstream/recgpt.db")

    path_str = to_string(path)
    dir = Path.dirname(path_str)
    if dir != "" and not File.exists?(dir), do: File.mkdir_p!(dir)

    {:ok, Keyword.put(config, :database, path_str)}
  end
end
