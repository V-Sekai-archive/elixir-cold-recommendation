# Ignore pattern_match in Application
[{"lib/recgpt/application.ex", :pattern_match},
 # traced_tensor_fn in run/1
 {"mix/tasks/recgpt.trace_predict.ex", :no_return},
 # Stream.transform typing limitation: success typing vs Enumerable contract
 {"lib/recgpt/eval.ex", :no_return},
 {"lib/recgpt/eval.ex", :call},
 # Defensive clause for non-binary; all current call sites pass binary
 {"lib/recgpt/steam/canonical_item_text.ex", :pattern_match_cov}]
