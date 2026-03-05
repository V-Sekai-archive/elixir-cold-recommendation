# Nsight Systems tracing for RecGPT recommender

Use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) to profile the recommender pipeline and correlate CPU phases with GPU kernels.

## Running a trace

### Ad-hoc test (recommend for 3 contexts)

```bash
mix recgpt.ad_hoc_test --profile --fixture data/steam/fixture.json --ckpt data/fuxi_ckpt_export
```

Writes `recgpt_adhoc_<timestamp>.nsys-rep`. Open in Nsight Systems GUI (`nsys-ui` or Nsight Systems).

### Trace predict (warm recommend timing + stats)

```bash
mix recgpt.trace_predict --profile --context "0,1" --top-k 10
```

Writes `recgpt_trace_<timestamp>.nsys-rep`. Focuses on warm recommend after JIT compile.

## NVTX markers

When libnvToolsExt is available, these markers annotate the timeline:

| Range                 | Location                                                             |
| --------------------- | -------------------------------------------------------------------- |
| `single_forward`      | One model forward (logits for last 4 positions)                     |
| `beam_search_step_0` | Beam step 0 (slice logits, trie gather, top_k)                      |
| `beam_search_step_1..3` | Beam steps 1–3 (slice logits_4, trie, top_k; no extra forwards)  |
| `mtp_forward`        | MTP path: one forward then score-all-items (no beam steps)          |
| `decode_sync`        | Final CPU sync (`Nx.to_flat_list`, `backend_transfer`)              |

NVTX uses `dlopen` at runtime; if libnvToolsExt is not found, markers no-op.

## What to look for

| Area                | Question                                                       | Expected                                        |
| ------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| **Kernel time**     | Is inference (GEMM/dot) dominant?                              | Yes — one forward, then beam steps or MTP score |
| **Step 0 vs 1–3**   | Beam: one forward; steps 0–3 slice precomputed logits           | No extra model forwards; trie gather + top_k   |
| **GPU utilization** | Gaps between kernel launches?                                  | Possible CPU-side work between steps            |
| **Sync points**     | Where does CPU wait on GPU?                                    | `Nx.to_flat_list`, `Nx.backend_transfer` at end |
| **Memory**          | H2D/D2H copies?                                                | Context tokens, final item_ids/scores           |

## Baseline trace findings (RTX 4090, 12-layer)

From a typical run (`mix recgpt.ad_hoc_test --profile`):

- **CUDA API**: `cuCtxSetCurrent`, `cuMemHostAlloc`, `cuMemcpyDtoHAsync_v2`, `cuStreamSynchronize` dominate API time.
- **NVTX (XLA)**: XlaCompile, XlaAutotunerCompilation, XlaCompileGpuAsm (gemm_fusion_dot) appear in nvtx_sum — JIT compile and GEMM kernels.
- **Stats**: `--stats=true` generates `.sqlite` with nvtx_sum, cuda_api_sum, etc.

Use these to guide Phase 3 improvements: multi-context batching, BF16, kernel fusion, etc.
