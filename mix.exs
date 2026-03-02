defmodule RecGPT.MixProject do
  @moduledoc """
  Single top-level Mix project for RecGPT.
  RecGPT Elixir library: FSQ, embeddings (MPNet/Bumblebee), training data pipeline. No GenServer; use from any app.
  """
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :recgpt,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      dialyzer: [
        plt_add_apps: [:mix],
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        ignore_warnings: ".dialyzer_ignore.exs"
      ],
      description:
        "RecGPT library: FSQ, text embeddings (MPNet), training batches and loss. Depends on Bumblebee."
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support", "mix"]
  defp elixirc_paths(_), do: ["lib", "mix"]

  def application do
    [
      extra_applications: [:logger],
      mod: {RecGPT.Application, []}
    ]
  end

  defp deps do
    [
      {:grpc, "~> 0.11"},
      {:protobuf, "~> 0.14"},
      {:nx, "~> 0.11", override: true},
      {:exla, "~> 0.10"},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: "main"},
      {:jason, "~> 1.4"},
      {:npy, "~> 0.1.2"},
      {:unpickler, "~> 0.1"},
      {:unzip, "~> 0.13"},
      {:req, "~> 0.5"},
      {:ecto_sqlite3, "~> 0.14"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 0.6", only: :test},
      {:benchee, "~> 1.3", only: :dev}
    ]
  end
end
