# Test config: disable checkpoint SHA256 verification so stub exports (e.g. CheckpointExportTest)
# can be written and loaded without matching the production checkpoint hash.
import Config

config :recgpt, :ckpt_expected_sha256, nil

# Use EXLA.Backend for tests with GPU acceleration
# NOTE: Changed from BinaryBackend to EXLA for performance testing
config :nx, default_backend: EXLA.Backend
config :nx, :default_defn_options, compiler: EXLA

# Use CUDA client for GPU acceleration
config :exla, :default_client, :cuda
