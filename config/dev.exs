# Dev: EXLA uses config :exla, :default_client (:cuda or :host). GPU is used when client is :cuda.
import Config

# BF16 (Tensor Cores) for faster inference.
config :recgpt, :inference_dtype, {:bf, 16}

# Context cache: warm at load so context=[0] (and empty) hit cache in trace_predict.
config :recgpt, :context_cache_warm_list, [[] , [0]]
config :recgpt, :context_cache_warm_batch_size, 4
