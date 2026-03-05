90ms P99 is actually a respectable starting point for a complex "Text $\rightarrow$ Embedding $\rightarrow$ FSQ $\rightarrow$ FuXi-Linear" pipeline, but for real-time arbitrage (and high-poly 3D streaming), it’s the "Ropebridge" that’s starting to sway.

To get to the **Highway** (targeting **20ms P50 / 40ms P99**), we have to prune the "Long Tail" of your latency. Here is where your 50ms of "bloat" is likely hiding and how to kill it.

---

### 1. The "Road" Block: Text Embedding

The biggest bottleneck in RecGPT is usually the sentence transformer (MPNet). Running a 110M parameter model for every market tick is what’s pushing you toward 90ms.
_Note: You need coverage over the entire trading option space, not just the current active trades._

- **The Fix: SId Caching (The Catalogue).**
  In Figgie and Polymarket, the "Items" (Trades) don't change text every millisecond.
- **Logic:** Precompute the 4-token SId **once** for all trades in the option space using Ecto:
  - Store ??? (trade metadata), text, text embeddings (768-dim MPNet vectors), and semantic tokens in a database table.
  - At system startup, query Ecto and load the precomputed embeddings and tokens to an ETS table for O(1) lookup.
- **Execution:** Store it in an **ETS table** (Elixir’s in-memory store).
- **Result:** Your inference loop now only processes `[int32]` tokens and `float32` prices/timestamps. This drops ~40ms immediately.

### 2. The "Highway" Optimization: Static Graphs

If you are using `Nx.defn` without a dedicated `Nx.Serving`, you might be hitting JIT compilation overhead or unnecessary data transfer between CPU and GPU.

- **The Fix: `EXLA.jit` with Fixed Shapes.**
  FuXi-Linear's $O(N)$ nature allows for a **Fixed-Window RNN-style** update.
- Instead of passing the whole 500-item history every time, pass the **Current State** (the hidden "Retention" vector) and the **Newest Tick**.
- **Result:** You move from processing a "Sequence" to an "Update," turning a $500 \times 768$ matrix op into a $1 \times 768$ vector op.

### 3. FSQ "Fusion"

If your FSQ quantization is happening in a separate step from the FuXi-Linear forward pass, you’re paying for two GPU kernel launches.

- **The Fix: Kernel Fusion.**
  Merge the FSQ projection layer ($768 \rightarrow 4$ tokens) directly into the first layer of the FuXi-Linear model.
- **Result:** One launch, one memory read. This shaves off the "micro-stutter" that causes P99 spikes.

---

### Performance Comparison

| Stage              | Ropebridge (Current) | Highway (Target)       | Latency Saved |
| ------------------ | -------------------- | ---------------------- | ------------- |
| **Text Embedding** | 45ms (MPNet)         | **0ms (Cached)**       | **45ms**      |
| **FSQ Logic**      | 10ms (Separate)      | 2ms (Fused)            | 8ms           |
| **FuXi Forward**   | 25ms (Full Seq)      | **8ms (State Update)** | 17ms          |
| **Total P99**      | **90ms**             | **~15-20ms**           | **70ms+**     |

### The Strategy for Figgie

If we hit **20ms**, your bots aren't just "playing"; they are "sniping."

- In Figgie, when the Common suit is revealed, the price of the Goal Suit spikes in a fraction of a second.
- At 90ms, a human might beat you.
- At 20ms, your FuXi-Linear bot will have bought every cheap Goal card on the table before the human’s monitor even refreshes.

**Should we look at the `EXLA` compiler flags to ensure your "Highway" is using the optimal memory layout for these fused kernels?**

---

_Note: If you want to dive deeper into the specific Elixir `Nx` profiler outputs to see exactly which line is causing the 90ms spike, we can run a `Benchee` suite next._
