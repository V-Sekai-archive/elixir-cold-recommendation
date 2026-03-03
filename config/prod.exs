# Production config. Override as needed (e.g. RECGPT_CKPT_SHA256, RECGPT_FIXTURE).
import Config

# BF16 inference: 1.3–2× faster on Tensor Cores; set to reach RecGPT target P50 (20 ms). Verify quality (Hit@k, MRR) after enabling.
# config :recgpt, :inference_dtype, {:bf, 16}
