# Constructor prediction test — local review candidate

**No paid model/credential/remote/Kaggle/gameplay entry is authorized or implemented.** This directory contains the fixed nested-schema constructor copy, an explicit grid-or-abstain contract, sealed execution, and an offline predictor/scorer. Read EXPERIMENT-CARD.md first.

## Evidence and isolation

The fixed selection is R10 B session `cd82-fb555c5d-b7azqbdf`: step 1 RIGHT for construction; immediate step 2 DOWN for holdout. Same session, level 0, NOT_FINISHED states and continuous frames were verified. Selection and scoring definitions were written before holdout inspection; no changed-case filtering. `RECORDED-PROVENANCE.json`, `recorded/SELECTION.json` and collection hash cross-check bind the exact recording. It is a retrospective chronological holdout, model-unseen in this invocation; operators/history have seen it. Existing R10 identity HOLD is unchanged.

The operator's offline preparation splits the already-known recording into:

- `recorded/prefix.jsonl`: only step 1; the sole constructor evidence.
- `recorded/query.json`: next action/step/session and hashes; not sent to construction.
- `recorded/heldout/outcome.json`: target, opened only by scoring after prediction binding.

The trusted preparation operator reads the recording to split it. This is distinct from the scoring process reading a target: model tool/predictor processes cannot access the target, its directory, the full recording, host files or environment. Construction tools see only sealed `/evidence/case.json`. Predictor processes receive only their source and current ABI arguments, without `/evidence`.

## Runtime path

1. Local loopback HTTP fixtures exercise the actual constructor and fixed nested work_item schema. The copied sealed evaluator now validates typed prediction/abstention shape on visible replay, without scoring accuracy.
2. A bounded worker performs construction. Only successful timely exit allows root source-freeze promotion; failed/unpromoted child files are forensic.
3. `predict` verifies a source/prefix-bound typed freeze, consumes the source-prefix-query identity in a registry, replays step 1 to reconstruct state, and invokes predict once for step 2. Next action is passed at this stage only.
4. Raw prediction, validated prediction and `BOUND.json` are written exclusively and fsynced. The command returns the binding SHA-256 externally.
5. `score` requires that external hash, rechecks source/prefix/query/raw/validated prediction, consumes scoring, and only then opens the target. No heldout assimilation, constructor feedback or retry exists.

All sealed RPC receipts, source/prefix/query/prediction/target hashes and chronological events are retained. Source files and prediction receipts are write-once; hash verification detects later tampering. A consumed source-prefix-query key cannot be reissued by changing output directory. Local fixture tests use their own disposable registries. This protects workflow integrity against accidental mutation/retry; a malicious host operator can rewrite its own files and is not a claimed adversarial trust boundary.

## Meaning of results

Executable typed construction, coverage, exact accuracy, full-grid cell accuracy and improvement over identity are separate fields. Abstention has coverage 0 and null prediction accuracy/differences. Empty/malformed commitments are invalid, not abstention. Changed-cell accuracy and false-change rate use the card's fixed denominators, with null for undefined. Exact identity on unchanged data is correct but does not beat identity. High unchanged-cell accuracy alone is no advantage.

An identity **fixture**, not a model, was run once through the offline commands on the selected real recorded transition (`local-recorded-identity/`). It predicted before target scoring and obtained 95.1171875% cell accuracy, zero exact match, 0/200 changed cells correct, and zero improvement over baseline. This is an arithmetic/isolation demonstration; no model competence evidence. Do not repeat that consumed demonstration.

## Verification command

From repository root, using a NEW basetemp:

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/constructor-prediction-test-20260916T233048Z/test_prediction.py --basetemp=analysis/constructor-prediction-test-20260916T233048Z/verification/reviewer
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/constructor-prediction-test-20260916T233048Z/check_candidate.py
```

Fixtures manufacture exact answers only to check the evaluator. Such source must not be mistaken for model-generated science. The protocol tests use real HTTP and sealed RPC, with failures and raw outputs retained under verification/.

## Offline use with an actual supplied candidate artifact

A supplied JSON artifact must contain complete `candidate_source` text and its SHA-256 in `source_digest`. This command rechecks it against the new typed visible contract and writes a NEW typed prefix-bound freeze; it never edits the provided artifact or repairs source. Old ABI-only artifacts may fail the new contract. A provided-artifact label does not independently certify model origin.

Run only when an artifact is supplied; replace paths, use new output directories and the exact binding printed by predict. All output paths must be under this new candidate directory. These commands are local and never call a model:

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/constructor-prediction-test-20260916T233048Z/prediction_test.py freeze-artifact --artifact /ABSOLUTE/PROVIDED-FROZEN.json --prefix analysis/constructor-prediction-test-20260916T233048Z/recorded/prefix.jsonl --output analysis/constructor-prediction-test-20260916T233048Z/provided-import
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/constructor-prediction-test-20260916T233048Z/prediction_test.py predict --frozen analysis/constructor-prediction-test-20260916T233048Z/provided-import/FROZEN.json --prefix analysis/constructor-prediction-test-20260916T233048Z/recorded/prefix.jsonl --query analysis/constructor-prediction-test-20260916T233048Z/recorded/query.json --output analysis/constructor-prediction-test-20260916T233048Z/provided-prediction
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B analysis/constructor-prediction-test-20260916T233048Z/prediction_test.py score --frozen analysis/constructor-prediction-test-20260916T233048Z/provided-import/FROZEN.json --prefix analysis/constructor-prediction-test-20260916T233048Z/recorded/prefix.jsonl --query analysis/constructor-prediction-test-20260916T233048Z/recorded/query.json --outcome analysis/constructor-prediction-test-20260916T233048Z/recorded/heldout/outcome.json --output analysis/constructor-prediction-test-20260916T233048Z/provided-prediction --binding-sha256 HASH_PRINTED_BY_PREDICT
```

No target feedback is returned to a constructor. The score is for operator review only. One consumed attempt is terminal even if invalid, abstained or failed. No paid run, source repair or future arm is authorized by a score. Independent reviewer owns approval; any future paid use requires separate review/card/PM release.
