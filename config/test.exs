# Test config: disable checkpoint SHA256 verification so stub exports (e.g. CheckpointExportTest)
# can be written and loaded without matching the production checkpoint hash.
import Config

config :recgpt, :ckpt_expected_sha256, nil
