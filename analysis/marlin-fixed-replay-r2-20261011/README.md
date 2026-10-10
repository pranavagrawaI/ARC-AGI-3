# Repaired fixed-capture Marlin diagnostic

Prepared for a second attempt, awaiting fresh launch authority. Kernel 138079507 v1 failed at runtime argument publication before any numerical/timing case because the original temporary draft-model view did not exist. Its artifacts are preserved in ../marlin-fixed-replay-20261011/readout/raw.

The repair reuses v3's exact prepare_draft_view and indexed_shards functions. It recreates the same config/index/tokenizer/weight-symlink view, checks its content-derived path matches the captured runtime argument, and records its config hash. No weights or server are loaded. The nine-case numerical/timing experiment, input tensors and reduction code stay unchanged.

One fresh GPU push, one RTX PRO 6000 and a 3600-second hard cap. Fresh user authority must be recorded in evidence/GO.json before push. Collection is read-only. No automatic retry or serving advancement.

Local check command:

```bash
uv run --no-project --with numpy --with nbformat --with requests python -B -m unittest discover -s analysis/marlin-fixed-replay-r2-20261011 -p 'test_*.py' -v
```

The mount/initialization/CUDA seam remains GPU-unqualified. Tests execute the exact metadata adapter against a fresh fixture and verify unchanged config identity, idempotence and symlinks rather than weight copies.
