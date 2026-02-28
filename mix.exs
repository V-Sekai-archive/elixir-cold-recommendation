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
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description:
        "RecGPT library: FSQ, text embeddings (MPNet), training batches and loss. Depends on Bumblebee.",
      test_coverage: [summary: [threshold: 85]]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib", "mix"]

  def application do
    [extra_applications: [:logger], mod: {RecGPT.Application, []}]
  end

  defp deps do
    [
      {:grpc, "~> 0.11"},
      {:protobuf, "~> 0.14"},
      {:nx, "~> 0.11", override: true},
      {:torchx, "~> 0.11"},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: "main"},
      {:jason, "~> 1.4"},
      {:npy, "~> 0.1.2"},
      {:unpickler, "~> 0.1"},
      {:unzip, "~> 0.13"},
      {:plug_cowboy, "~> 2.6"},
      {:ecto_sqlite3, "~> 0.14"},
      {:req, "~> 0.5"},
      {:rdf, "~> 2.0"},
      {:json_ld, "~> 1.0"},
      {:grax, "~> 0.6"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:propcheck, "~> 1.5", only: [:dev, :test]}
    ]
  end
end
