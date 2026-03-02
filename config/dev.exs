# Dev: use CUDA for inference when available (~100ms per Predict vs ~1.5s on CPU).
# If EXLA fails to load (no GPU), set RECGPT_EXLA_HOST=1 and restart.
import Config

if System.get_env("RECGPT_EXLA_HOST") != "1" do
  config :exla, :default_client, :cuda
end
