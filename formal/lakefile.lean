import Lake
open Lake DSL

-- SPDX-License-Identifier: MIT OR Apache-2.0
-- Formal model of the RecGPT semantic-ID codec that ships in this repo:
--   * FSQ mixed-radix index codec  (lib/recgpt/fsq.ex: codes_to_indices / indices_to_codes)
--   * trie-constrained 4-token decode (lib/recgpt/trie.ex, lib/recgpt/decode.ex)
-- Built on fire/plausible-witness-dag.

package «recgpt-codec-model» where

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_lib RecGptCodec where

lean_exe «recgpt-codec-sample» where
  root := `RecGptCodec
