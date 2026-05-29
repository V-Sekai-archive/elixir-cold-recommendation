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
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description:
        "RecGPT library: FSQ, text embeddings (MPNet), training batches and loss."
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
      {:nx, "~> 0.11", override: true},
      {:exla, "~> 0.11"},
      {:xla, "~> 0.10"},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: "main"},
      {:jason, "~> 1.4"},
      {:npy, "~> 0.1.2"},
      {:unpickler, "~> 0.1"},
      {:unzip, "~> 0.13"},
      {:req, "~> 0.5"},
      {:nimble_csv, "~> 1.2"},
      {:explorer, "~> 0.9"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.2", only: [:dev, :test]},
      {:propcheck, "~> 1.5", only: [:test]}
    ]
  end
end
