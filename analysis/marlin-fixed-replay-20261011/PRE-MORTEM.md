# Fixed-capture replay pre-mortem

The failure scene is another GPU bill with no usable speed observation, or a misleading improvement caused by changing the inputs or acceptance test.

- v3 stopped at the first finite numerical rejection before timing. Preserve its unchanged verdict while timing all finite cases diagnostically; never interpret this as serving acceptance.
- An unpinned notebook-output attachment could mount a newer source version. Check kernel 138024107 is still v1 before/after collection, and hash every required file against saved v3 hashes in the GPU notebook.
- Notebook inputs might not include the expected capture path. Search for exactly one capture/cases.jsonl and fail before installing/replaying if its hash differs. This mount seam cannot be exercised locally against Kaggle.
- Runtime initialization may need model config despite no server being started. Retain the original model attachments and server-args initialization, without loading model weights.
- Two GPU graphs could share a live output tensor and manufacture agreement. Allocate a separate control callable/graph and clone each synchronized output before the next replay. Repeat fixture tests prove copies survive later output overwrites.
- The collector previously retried collection when its audit interpreter lacked NumPy. Use the existing NumPy interpreter in a subprocess and make a failed audit terminal after successful download.
- The quota endpoint now reports 21600 seconds allowed but more already used. Historical 108000-second checks are insufficient. Preserve the typed receipt, require reconciliation with authoritative available quota, and fail before the exclusive push if unresolved.
- A rare atomic order may not appear in 8 repetitions. Report an observed envelope only. Do not use it as a universal arithmetic bound.
- Diagnostic references and extra control graphs affect allocated memory. Numerical probes precede clean original-input warmed timing; measured allocations describe this replay only. Serving residency remains untested.
- Timeout or later-case failure can lose intermediate evidence. Write each case/result immediately and hash outputs in notebook finally; completed cases remain useful locators but incomplete all-case screening cannot advance anything.

Accepted residual risks: CUDA/JIT behavior and mounted-output layout are not locally executable; the original GPU run qualified the shared package/candidate. The new repetition/collection logic receives CPU behavioral tests, and the one-attempt run preserves any remaining failure.
