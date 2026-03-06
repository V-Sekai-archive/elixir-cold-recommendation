defmodule Mix.Tasks.Recgpt.FiggieSimulate do
  @moduledoc """
  Simulates Figgie games and generates training data for RecGPT arbitrage trading.

  Usage:
    mix recgpt.figgie_simulate [options]

  Options:
    --games, -g    Number of games to simulate (default: 1000)
    --output, -o   Output file for fixture data (default: priv/figgie_fixture.json)
  """

  use Mix.Task
  alias RecGPT.Figgie.DataFetcher

  @impl Mix.Task
  def run(args) do
    {opts, _} =
      OptionParser.parse!(args,
        aliases: [g: :games, o: :output],
        switches: [games: :integer, output: :string]
      )

    games = opts[:games] || 1000
    output = opts[:output] || "priv/figgie_fixture.json"

    Mix.shell().info("Simulating #{games} Figgie games...")

    # Generate training data
    games_data = RecGPT.Figgie.DataFetcher.fetch_training_data(games)
    Mix.shell().info("Generated data for #{length(games_data)} games")

    # Convert to fixture format
    fixture_data = RecGPT.Figgie.DataFetcher.to_fixture_data(games_data)
    Mix.shell().info("Converted to fixture format")

    # Write to file
    json_data = Jason.encode!(fixture_data, pretty: true)
    File.write!(output, json_data)

    Mix.shell().info("Fixture data written to #{output}")
    Mix.shell().info("Run training with: mix recgpt.pretrain --fixture #{output}")
  end
end
