#!/usr/bin/env elixir

# Create minimal stub fixture and checkpoint for baseline latency testing
# This allows testing the recommend path without real data

Mix.shell().info("Creating stub fixture and checkpoint for baseline testing...")

# Ensure applications are started
Application.ensure_all_started(:nx)
Application.ensure_all_started(:jason)

# Create stub checkpoint
ckpt_dir = Path.expand("data/test_ckpt_export", File.cwd!())
Mix.shell().info("Writing stub checkpoint to #{ckpt_dir}...")

RecGPT.TestSupport.FrozenHelpers.write_stub_ckpt!(ckpt_dir)

# Create stub fixture
fixture_path = Path.expand("data/test_fixture.json", File.cwd!())
Mix.shell().info("Writing stub fixture to #{fixture_path}...")

# Create minimal fixture with 2 items
token_id_list = [
  [100, 200, 300, 400],  # Item 0 tokens
  [101, 201, 301, 401]   # Item 1 tokens
]

fixture = %{
  "token_id_list" => token_id_list,
  "num_items" => length(token_id_list)
}

File.write!(fixture_path, Jason.encode!(fixture, pretty: true))

Mix.shell().info("")
Mix.shell().info("✓ Created test data:")
Mix.shell().info("  Fixture: #{fixture_path}")
Mix.shell().info("  Checkpoint: #{ckpt_dir}")
Mix.shell().info("")
Mix.shell().info("Now run:")
Mix.shell().info("  mix recgpt.trace_predict --fixture #{fixture_path} --ckpt #{ckpt_dir} --runs 20")
