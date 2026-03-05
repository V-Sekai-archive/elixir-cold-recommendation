#!/usr/bin/env elixir

# Direct script to generate Figgie training data

IO.puts("Generating Figgie training data...")

# Generate training data
games_data = RecGPT.Figgie.DataFetcher.fetch_training_data(100)
IO.puts("Generated data for #{length(games_data)} games")

# Convert to fixture format
fixture_data = RecGPT.Figgie.DataFetcher.to_fixture_data(games_data)
IO.puts("Converted to fixture format")

# Write to file
json_data = Jason.encode!(fixture_data, pretty: true)
File.write!("../priv/figgie_fixture.json", json_data)

IO.puts("Fixture data written to ../priv/figgie_fixture.json")
IO.puts("Run training with: mix recgpt.pretrain --fixture ../priv/figgie_fixture.json")