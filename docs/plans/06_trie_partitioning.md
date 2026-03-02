# Plan: Trie partitioning for 13M

**Status:** Todo | **Est. gain:** Enables 13M; partition load may add 10–50 ms | **Profit:** 1.3 | **Effort:** High | **Gain:** Enables 13M

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

Partition trie by first token (or coarse category) so each shard fits in memory and can be loaded on demand.

---

## Dependency

Works with dense or sparse trie; [sparse trie](05_sparse_trie.md) makes per-partition size manageable.

---

## Changes

- [lib/recgpt/trie.ex](../lib/recgpt/trie.ex): `build_partitioned/2` returns `%{first_token => sub_trie}`. Each sub-trie handles items whose first token matches.
- [lib/recgpt/decode.ex](../lib/recgpt/decode.ex): Step 0 constrains to first tokens; select partition by chosen token; steps 1–3 use that partition only.
- [lib/recgpt/serve.ex](../lib/recgpt/serve.ex): Load partitions lazily or keep N hot partitions in memory.

---

## When to use

If single sparse trie still too large after [Plan 5](05_sparse_trie.md).
