# Rope Bridge Analogy: From ZeroMQ ZGuide Chapter 7

This doc documents the **rope bridge vs busy road vs multilane highway** analogy and its origin in the [ZeroMQ ZGuide, Chapter 7 — Advanced Architecture using ZeroMQ](https://zguide.zeromq.org/docs/chapter7/).

---

## Origin: MOPED and the Rope Across the Gorge

The ZGuide’s MOPED pattern (Message-Oriented Pattern for Elastic Design) uses a physical metaphor:

> Start by choosing the core problem that you are going to solve. Ignore anything that's not essential to that problem: you will add it in later. **The problem should be an end-to-end problem: the rope across the gorge.**
>
> ...
>
> Your goal is not to define a _real_ architecture, but to **throw a rope across the gorge to bootstrap your process**. We make the architecture successfully more complete and realistic over time: e.g., adding multiple workers, adding client and worker APIs, handling failures, and so on.

The idea: build the smallest working end-to-end path first, then grow it. Don’t overbuild before you know the route works.

---

## Our Three-Stage Extension: Rope Bridge → Busy Road → Multilane Highway

We extend this into a **three-stage progression**:

| Stage                    | Metaphor              | Meaning                                                                                              | Example (this codebase)                                                                                                                                                   |
| ------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Rope bridge**       | Minimal crossing      | Bootstrap: smallest end-to-end path that works. One lane, fragile but sufficient to prove the route. | Single-rank eval (`RecGPT.Eval`), paper trading ([60 Rope bridge](60_rope_bridge_market_analytics_plan.md)), first step on Steam ([24 First step](24_first_step_plan.md)) |
| **2. Busy road**         | Improved capacity     | Next phase: more traffic, some hardening, still a single lane but robust.                            | Live trading with small capital, one rank with pipelining                                                                                                                 |
| **3. Multilane highway** | Scaled infrastructure | Full scale: SPMD, sharding, high throughput.                                                         | Multi-rank eval, distributed serving, scaled trading                                                                                                                      |

The sequence is strict: **rope bridge first**. If the rope bridge fails, you never reach the road or highway.

---

## Where We Use It

- **[25 MVP guard rails](25_mvp_guard_rails.md)** — Tombstones: no multi-rank/sharding until the minimal loop is closed. “Keep the rope bridge on track.”
- **[60 Rope bridge market analytics](60_rope_bridge_market_analytics_plan.md)** — Paper trading must survive (no bankruptcy) before progressing to busy-road and multilane-highway stages.
- **[RecGPT.Eval](lib/recgpt/eval.ex)** — “We build a rope bridge across the chasm before a road or a busy highway”: eval is single-rank, one wavefront; SPMD and scaling come later.
- **[24 First step plan](24_first_step_plan.md)** — Steam as test vectors to close the minimal loop (the rope bridge) before other catalogs or integrations.

---

## See also

- [ZeroMQ ZGuide, Chapter 7](https://zguide.zeromq.org/docs/chapter7/) — MOPED, “rope across the gorge”
- [25 MVP guard rails](25_mvp_guard_rails.md) — Tombstones for rope-bridge discipline
- [60 Rope bridge market analytics](60_rope_bridge_market_analytics_plan.md) — Paper trading → busy road → highway
