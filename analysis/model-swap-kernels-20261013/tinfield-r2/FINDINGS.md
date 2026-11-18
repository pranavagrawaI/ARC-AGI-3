# Tinfield r2: not launched; it would have failed in assembly again

Before the push, the notebook's own `assemble()` was run against the real saved indices, using header-only shards with real tensor names and the live Kaggle core file listing (`assembly_exercise.py`). r2 fails with `AssertionError: .../model-00001-of-00010.safetensors`.

Cause: the NVFP4 export stores PLE as one unsharded tensor, `layers.1.ple.ple_embedding.ngram_embedding.weight`, which is all of `model-00001` (102.4 GB). The Kaggle core instance omits that file, but its index still names it. The incumbent uses 128 `ngram_embedding.shard_N.weight` keys instead. r2's `is_ple` matched only `.shard_` names, so the unsharded key fell through to the core-shard loop's `is_file` assertion. v1 and r2 both checked their fix against an index model rather than against the files that are actually mounted.

No push, no GPU, no slot claim, no queue entry.

A minimal fix, unpushed, is in `candidate-r3/`: it changes only cell 4 and drops the unsharded key before the PLE inventory. It passes the assembly exercise (296,474 tensors, nothing missing), the schema and compile checks, the identical TRUE_SUBMISSION bodies check and `incumbent/check.py --without b1`. The incumbent's Kaggle PLE comes as 22 pure `model-ple-*` shards, so it is linked, not copied (no disk risk). Launching it needs Pranav's authorisation.

Still unverifiable before a GPU run: SGLang ModelOpt NVFP4 load with the Intel config plus the swapped quantization_config, and the linked AutoRound `quantization_config.json` sidecar. The config's quantization_config should take precedence over that sidecar.
