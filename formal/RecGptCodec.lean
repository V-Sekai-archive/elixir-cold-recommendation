import PlausibleWitnessDag

/-! # Formal model of the RecGPT semantic-ID codec (as shipped in this repo)

Models the two pieces of `lib/recgpt/` that decode must get exactly right, at the **real** scale:

* **FSQ mixed-radix index codec** — `lib/recgpt/fsq.ex`. `codes_to_indices` maps a 5-dim per-position
  FSQ code `(c0,c1,c2,c3,c4)` with `levels = [8,8,8,6,5]` to a single token index via the basis
  `[1,8,64,512,3072]` (= cumprod of `[1,8,8,8,6]`): `index = c0 + c1*8 + c2*64 + c3*512 + c4*3072`.
  `indices_to_codes` inverts it as `c_i = (index / basis_i) % levels_i`. Vocabulary is `15360`
  (`= 8*8*8*6*5`); padding id is `15360`.

* **Trie-constrained 4-token decode** — `lib/recgpt/trie.ex` / `decode.ex`. An item is 4 tokens; the
  catalog trie maps a full 4-token path to `item_id`, and `valid_next_tokens` gives the legal
  continuations at each depth so beam search only ever emits real catalog items.

The codec facts are proved by `omega` (linear + div/mod by the literal bases — symbolic, no
enumeration, so they hold at the real `15360`-vocab scale). The decode is certified by a
`fire/plausible-witness-dag` witness: a deterministic trie walk reaches a catalog item's 4-token path;
a shallow ladder rung (budget `< 4`) budget-hits, a deeper rung resolves.
-/

namespace RecGptCodec

open PlausibleWitnessDag

/-- Tokens per item (`RecGPT.FSQ.seq_len` / `RecGPT.Trie.seq_len`). -/
def seqLen : Nat := 4

/-- FSQ codebook size `= 8*8*8*6*5` (`RecGPT.FSQ.vocab_size`). -/
def vocab : Nat := 15360

/-- Padding id (`RecGPT.FSQ.padding_id`), one past the content range. -/
def paddingId : Nat := 15360

/-- The mixed-radix index for a 5-dim FSQ code, with basis `[1,8,64,512,3072]`
    (`RecGPT.FSQ.codes_to_indices`). -/
def fsqIndex (c0 c1 c2 c3 c4 : Nat) : Nat :=
  c0 + c1 * 8 + c2 * 64 + c3 * 512 + c4 * 3072

/-- Every valid FSQ code lands inside the vocabulary (`< 15360`); padding is disjoint. -/
theorem fsq_bound
    (c0 c1 c2 c3 c4 : Nat)
    (h0 : c0 < 8) (h1 : c1 < 8) (h2 : c2 < 8) (h3 : c3 < 6) (_h4 : c4 < 5) :
    fsqIndex c0 c1 c2 c3 c4 < vocab ∧ fsqIndex c0 c1 c2 c3 c4 < paddingId := by
  unfold fsqIndex vocab paddingId; omega

/-- `indices_to_codes ∘ codes_to_indices = id`: each digit is recovered by `(idx / basis_i) % levels_i`. -/
theorem fsq_roundtrip
    (c0 c1 c2 c3 c4 : Nat)
    (h0 : c0 < 8) (h1 : c1 < 8) (h2 : c2 < 8) (h3 : c3 < 6) (_h4 : c4 < 5) :
    fsqIndex c0 c1 c2 c3 c4 % 8 = c0 ∧
    fsqIndex c0 c1 c2 c3 c4 / 8 % 8 = c1 ∧
    fsqIndex c0 c1 c2 c3 c4 / 64 % 8 = c2 ∧
    fsqIndex c0 c1 c2 c3 c4 / 512 % 6 = c3 ∧
    fsqIndex c0 c1 c2 c3 c4 / 3072 = c4 := by
  unfold fsqIndex; omega

/-- The index codec is injective on valid codes — distinct FSQ codes never share a token id. -/
theorem fsq_injective
    (c0 c1 c2 c3 c4 d0 d1 d2 d3 d4 : Nat)
    (hc0 : c0 < 8) (hc1 : c1 < 8) (hc2 : c2 < 8) (hc3 : c3 < 6) (_hc4 : c4 < 5)
    (hd0 : d0 < 8) (hd1 : d1 < 8) (hd2 : d2 < 8) (hd3 : d3 < 6) (_hd4 : d4 < 5)
    (h : fsqIndex c0 c1 c2 c3 c4 = fsqIndex d0 d1 d2 d3 d4) :
    c0 = d0 ∧ c1 = d1 ∧ c2 = d2 ∧ c3 = d3 ∧ c4 = d4 := by
  unfold fsqIndex at h; omega

/-! ## Trie-constrained 4-token decode as a plausible-witness-dag witness -/

/-- A tiny catalog of 4-token items (token ids are real FSQ indices `< 15360`; some share prefixes so
    the trie is non-trivial). Item id = list index, mirroring `RecGPT.Trie` leaves. -/
def catalog : List (List Nat) :=
  [[3, 8, 64, 100], [3, 8, 70, 5], [1, 2, 3, 4], [3, 8, 64, 200]]

/-- `RecGPT.Trie.valid_next_tokens`: legal continuations of `pre` toward some catalog item. -/
def validNext (pre : List Nat) : List Nat :=
  (catalog.filter (fun s => s.take pre.length == pre && pre.length < s.length)
    |>.map (fun s => s[pre.length]!)).eraseDups

/-- `RecGPT.Trie.lookup`: the item id of a complete 4-token path, if any. -/
def lookup (seq : List Nat) : Option Nat := catalog.findIdx? (· == seq)

/-- Deterministic trie-constrained decode of `itemId`, one token per step, budgeted by `steps`.

Emits the item's 4-token path iff every step stayed inside the trie and depth `seqLen` is reached; needs
`steps ≥ seqLen` to finish. -/
def decodeWalk (steps : Nat) (itemId : Nat) : Option (List Nat) := Id.run do
  match catalog[itemId]? with
  | none => pure none
  | some seq =>
    let mut pre : List Nat := []
    let mut ok := true
    for slot in List.range seqLen do
      if ok && slot < steps then
        let tok := seq[slot]!
        if (validNext pre).contains tok then
          pre := pre ++ [tok]
        else
          ok := false
      else
        ok := false
    if ok && pre == seq && lookup seq == some itemId then some pre else none

/-- Two-rung ladder: L0's budget (3) is below `seqLen = 4`, so it budget-hits; L1 (4) resolves. -/
def recgptLevels : Array Level := #[
  { idx := 0, walkSteps := 3, finBound := 256, numInst := 200 },
  { idx := 1, walkSteps := 4, finBound := 256, numInst := 200 }]

/-- Candidate: a witness iff it names `itemId` and the budgeted trie walk reaches it. -/
def candidate (itemId : Nat) (lvl : Level) (c : Nat) : Bool :=
  c == itemId && (decodeWalk lvl.walkSteps itemId).isSome

/-- Deterministic read-back: the decoded 4-token path when reachable, else a budget-hit flag. -/
def readback (itemId : Nat) (steps : Nat) : Readback (List Nat) :=
  match decodeWalk steps itemId with
  | some seq => { value := seq, found := true, witnessIdx := itemId, budgetHit := false }
  | none => { value := [], found := false, witnessIdx := 0, budgetHit := (decodeWalk 100 itemId).isSome }

/-- Resolve the trie-constrained decode of one catalog item through the generic ladder. -/
def runSample (itemId : Nat := 0) : IO (List Nat × Nat × TraceEntry) :=
  resolve s!"RecGPT trie-constrained 4-token decode of item {itemId}" (candidate itemId)
    (readback itemId) recgptLevels

/-- Executable smoke test: L0 budget-hits, L1 resolves the item to a real catalog leaf. -/
def runSmokeTest : IO Unit := do
  let (seq, lvl, trace) ← runSample 0
  IO.println s!"resolved level: L{lvl}"
  IO.println s!"decoded tokens: {seq}"
  IO.println s!"trace: {repr trace}"
  if lvl != 1 || lookup seq != some 0 then
    throw <| IO.userError "RecGPT decode sample did not resolve at L1"
  IO.println "FSQ index codec certified (fsq_bound / fsq_roundtrip / fsq_injective by omega)"

end RecGptCodec

/-- Lake executables need a top-level `main`. -/
def main (_args : List String) : IO Unit :=
  RecGptCodec.runSmokeTest
