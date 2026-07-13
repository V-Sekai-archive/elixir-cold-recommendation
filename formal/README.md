# formal/ — Lean model of the RecGPT semantic-ID codec

A small Lean 4 model of the parts of `lib/recgpt/` that decode must get exactly right, built on
[`fire/plausible-witness-dag`](https://github.com/fire/plausible-witness-dag). It certifies the
invariants the Elixir/Nx code relies on, at the **real** scale.

- **FSQ mixed-radix index codec** — `lib/recgpt/fsq.ex` (`codes_to_indices` / `indices_to_codes`).
  `RecGptCodec.lean` proves by `omega` (symbolic, no enumeration, so it holds at the real 15360-vocab):
  - `fsq_bound` — every valid code maps below `vocab = 15360`; padding (`15360`) is disjoint.
  - `fsq_roundtrip` — `(idx / basis_i) % levels_i` recovers each digit (`indices_to_codes` inverts
    `codes_to_indices`) for `levels = [8,8,8,6,5]`, `basis = [1,8,64,512,3072]`.
  - `fsq_injective` — distinct FSQ codes never collide on a token id.
- **Trie-constrained 4-token decode** — `lib/recgpt/trie.ex` / `decode.ex`. A `plausible-witness-dag`
  witness: a deterministic trie walk decodes a catalog item's 4-token path; a shallow ladder rung
  (budget `< 4`) budget-hits, a deeper rung resolves to a real catalog leaf.

## Build & run

```bash
cd formal
lake update      # fetches plausible-witness-dag + plausible (Lean 4.30.0)
lake build       # type-checks the omega proofs
lake exe recgpt-codec-sample
# -> resolved level: L1 ; decoded tokens: [3, 8, 64, 100] ; codec certified
```
