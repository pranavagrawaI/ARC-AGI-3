# Prepared Marlin reduction screen

Local preparation is complete. GPU behavior is not qualified; no launch, upload or speedup result exists.

The exact qualified SGLang wheel is the authority. `candidate.patch` makes an optional target GPTQ reduction override and graph capture hooks; the local reference checkout and30 frozen v3 files are unchanged. The prepared notebook captures baseline inputs, then stops the serving process and runs clean paired expert replays. It does not run treatment serving or gameplay.

Artifacts:

- `EXPERIMENT-CARD.md`: fixed nine-case contrast, numerical/performance gates, one-hour GPU cap, invalidity and authority.
- `PLAN.md` and `PRE-MORTEM.md`: design, failure paths and residual uncertainty.
- `prepare.py`, `patched/`, `candidate.patch`, `patch-manifest.json`: exact-source scoped candidate.
- `marlin_capture.py`: device snapshot nodes in captured graphs, flushed only after real armed target verification replays; preserved graph keys, routing and weight hashes.
- `reference.py`: independent W4 inverse and dequantization, arithmetic and paired timing gates.
- `benchmark.py`: runtime pack/scale sentinel, actual reduction-call receipts, shared captured tensors, clean AB/BA timings, and separate scratch-NaN qualification graph with changed/restored inputs.
- `kernel/`: offline built notebook and private ARC-attached metadata. The benchmark initializes the same serving argument context, verifies installed patch hashes and disables TF32 for the FP32 oracle.
- `audit_capture.py`: reuses the qualified raw v3 block auditor and recomputes numerical/timing verdicts from raw NPZ arrays and paired intervals.
- `evidence/local-tests.log`, `local-verification.json`, `risk-final.json`, `REVIEW.md`: local evidence and explicit GPU coverage limits.

Rebuild and check locally:

```bash
python3 analysis/marlin-reduction-replay-v2-20261010/build_notebook.py
python3 -m unittest discover -s analysis/marlin-reduction-replay-v2-20261010 -p 'test_*.py' -v
```

After a separately authorized single GPU attempt, collect all outputs before interpreting them and run:

```bash
python3 analysis/marlin-reduction-replay-v2-20261010/audit_capture.py \
  analysis/marlin-reduction-replay-v2-20261010/readout/raw/marlin-replay \
  --output analysis/marlin-reduction-replay-v2-20261010/readout/AUDIT.json
```

Launch boundary: the notebook is prepared, but no launch call has been made. A fresh launch instruction must supply authority. At that point verify this committed prepared manifest, create the new identity's exclusive attempt receipt, perform the existing typed quota/idle-slot admission and one-attempt Kaggle push with3600-second timeout, and arm a read-only collector. Do not reuse a v3 launch identity, alter the card after launch, upload a dataset, silently retry, or advance serving automatically. The original package/replay/model attachments are reused, so this candidate requires no new dataset upload.

A positive screen means prepare the separately matched S10 serving comparison. The per-case3% screen and within-session paired interval cannot establish the proposed10% whole-batch serving improvement. Full model graph residency and original KV capacity must be tested there; arithmetic tolerances do not establish output-token/gameplay equivalence.
