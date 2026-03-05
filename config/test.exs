# Test config: disable checkpoint SHA256 verification so stub exports (e.g. CheckpointExportTest)
# can be written and loaded without matching the production checkpoint hash.
import Config

config :recgpt, :ckpt_expected_sha256, nil

# Use Nx.BinaryBackend for tests so they run without EXLA/CUDA (e.g. WSL, CI).
# EXLA has no precompiled XLA for Windows; run tests in WSL: wsl -e bash -c "cd /mnt/c/path/to/elixir-recgpt && mix test"
config :nx, default_backend: Nx.BinaryBackend

# For tests, use EXLA with host client to allow defn functions.
config :exla, :default_client, :host
