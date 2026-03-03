# Nsight Systems tracing for RecGPT recommender

Use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) to profile the recommender pipeline and correlate CPU phases with GPU kernels.

## Running a trace

### Ad-hoc test (recommend for 3 contexts)

```bash
mix recgpt.ad_hoc_test --profile --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export
```

Writes `recgpt_adhoc_<timestamp>.nsys-rep`. Open in Nsight Systems GUI (`nsys-ui` or Nsight Systems).

### Trace predict (warm recommend timing + stats)

```bash
mix recgpt.trace_predict --profile --context "0,1" --top-k 10
```

Writes `recgpt_trace_<timestamp>.nsys-rep`. Focuses on warm recommend after JIT compile.

## NVTX markers

When libnvToolsExt is available, these markers annotate the timeline:

| Range | Location |
|-------|----------|
| `beam_search_step_0` | First forward (context only, full sequence) |
| `beam_search_step_1` | Incremental step 1 (batch = beam_width) |
| `beam_search_step_2` | Incremental step 2 |
| `beam_search_step_3` | Incremental step 3 (item_at_leaf, returns item_ids) |
| `forward_with_cache` | Full forward_with_cache (step 0) |
| `forward_incremental` | Incremental forward (steps 1–3) |
| `decode_sync` | Final CPU sync (`Nx.to_flat_list`, `backend_transfer`) |

NVTX uses `dlopen` at runtime; if libnvToolsExt is not found, markers no-op.

## What to look for

| Area | Question | Expected |
|------|----------|----------|
| **Kernel time** | Is inference (GEMM/dot) dominant? | Yes — 4 forwards, mostly attention/dense |
| **Step 0 vs 1–3** | Is step 0 (full context forward) much larger than incremental? | Step 0: full sequence; 1–3: 1 token + cache |
| **GPU utilization** | Gaps between kernel launches? | Possible CPU-side work between steps |
| **Sync points** | Where does CPU wait on GPU? | `Nx.to_flat_list`, `Nx.backend_transfer` at end |
| **Memory** | H2D/D2H copies? | Context tokens, final item_ids/scores |

## Baseline trace findings (RTX 4090, 12-layer)

From a typical run (`mix recgpt.ad_hoc_test --profile`):

- **CUDA API**: `cuCtxSetCurrent`, `cuMemHostAlloc`, `cuMemcpyDtoHAsync_v2`, `cuStreamSynchronize` dominate API time.
- **NVTX (XLA)**: XlaCompile, XlaAutotunerCompilation, XlaCompileGpuAsm (gemm_fusion_dot) appear in nvtx_sum — JIT compile and GEMM kernels.
- **Stats**: `--stats=true` generates `.sqlite` with nvtx_sum, cuda_api_sum, etc.

Use these to guide Phase 3 improvements: multi-context batching, BF16, kernel fusion, etc.
