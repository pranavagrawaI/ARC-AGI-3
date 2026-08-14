# 0158 LS20 live run — semantic investigator interface

## Result

The repaired interface crossed the requested boundary: Qwen produced an
accepted grounding claim, produced an accepted prospective commitment, and the
host executed **one ARC action**. The public successor contradicted the
prediction, so the scored run terminated with **0 levels completed** and was
not resumed.

- Qualification: `artifacts/provider-qualification/20260812T134237Z-qwen36-semantic-interface`
- Run: `artifacts/live-ls20/20260812T134429Z-qwen36-semantic-interface`
- Run ID: `ef9283c2836045b495869ee6e3957b1a`
- Action: `1`, recorded in `issued-actions.jsonl`
- Provider crossings: 12 successful calls; no recorded 429
- Counted usage: 138,319 provider-reported input plus output tokens
- Scorecard: terminal `PAUSED_FOR_REVIEW`, non-resumable

## What changed

The model prompt now explains why evidence, claim, commitment, action, and
verdict are separated; describes the full phase lifecycle and non-oracular
verifier; assigns semantic judgments to Qwen and mechanical identifiers/schema
binding to the host; and includes exact claim, translation, and replacement
examples. The sandbox adds `current_frame()`. Claim submission automatically
binds the current observation, while commitment submission automatically binds
the accepted host `claim_id`. Failure packets record the actual phase-specific
callable surface.

The complete offline suite passed. A fresh call through the exact
Qwen/CoreWeave worker then passed claim and commitment qualification, including
an anchored translation effect. This proved only the interface path, not LS20
competence.

## Live evidence

Qwen claimed a 3x3 region at row 31, column 20 and supplied two alternative
regions. It selected legal action 1 and prospectively predicted an exact
replacement of the primary region. The host accepted both artifacts and issued
the action under commitment
`commitment-4e92da3fac2576c8c59f16a4ab58c95061303f9e742f82ed73bb6db61a6d5fc4`.

The successor contained 52 changed public cells, concentrated in rows 40–62
and columns 13–38. The asserted primary-region replacement did not occur. The
verifier returned `contradicted` with
`PRIMARY_REPLACEMENT_MISMATCH`, invalidated G0, and closed action authority.
This is evidence of a Qwen semantic grounding or effect-prediction failure,
not another interface or transport failure.

The immutable packet retains the legacy label `gemma_semantic`. A separate
`classification-correction.json` records the model-neutral class
`model_semantic` without modifying packet evidence, and the candidate source
now uses the model-neutral label.

## Preservation and verification

- The one-run approval is consumed and records `resumable: false`.
- All failure-packet checksums pass.
- All 20 files in the executed candidate manifest were hash-checked and copied
  to `candidate-source-snapshot`; `candidate-snapshot-verification.json`
  records the result.
- Public environment evidence records distinct pre-action and successor layer
  hashes.

The action-production objective is achieved. Gameplay success remains
`levels_completed >= 1`; this run does not establish it.
