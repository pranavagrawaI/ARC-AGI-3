# Independent exact-byte review — CONSTRUCTOR-PREDICTION-TEST

## Decision: PASS for the offline package only

No blocking defect found in the reviewed offline fixture/provided-artifact workflow. This does not approve a paid entry, which is neither implemented nor exercised, and does not establish model prediction competence. Existing R10 target-identity HOLD remains unchanged. No credentials, provider calls, gameplay, source changes, or remote mutations were performed. Writes were confined to this independent-review directory.

Exact candidate: `analysis/constructor-prediction-test-20260916T233048Z`.
Inventory SHA-256: `90073a670e3db0b0ada54fa544eada88d5030230e48ba1054401ba7f468238b4`.
Card SHA-256: `c34b75362b27d922338ab7fe90beeab035c413f859d10277e7003debd7006be1`.
All 42 inventory members, including the heldout target, match. `audit.json` preserves every input binding. Changed inputs invalidate this decision.

## Evidence and coverage

- **Real integration:** independently reran all 44 tests with the declared mapped-user wrapper: **44 passed in 29.76s**. Fresh reviewer basetemp and raw sealed receipts are retained. Tests traverse loopback HTTP, nested tool requests, constructor feedback/repair, sealed visible evaluation, supervised freeze promotion, frozen prediction and scoring. See `test_prediction.py:35`, `:167`, `:180`, and `:200`. The hard-kill test verifies that a child-written freeze is not promoted after incomplete exit.
- **Typed prediction:** `prediction_contract.py:4–26` requires exactly one branch with a full rectangular integer grid of the observed dimensions or explicit nonblank abstention. Bool, float, nonfinite, malformed, extra-key and multiple-branch cases fail. Visible replay and heldout prediction call the same validator (`engine/evaluate_visible.py:40`, `pipeline.py:140`). The nested source/work_item transport is inherited unchanged from the fixed interface. The only changed inherited files are the prompt contract and visible evaluator; source hashes were independently compared with their retained originals.
- **Isolation:** constructor receives the step-1 prefix only. Inspection uses its sealed prefix file; prediction receives source and ABI arguments in the chroot, without inspection evidence. Real tests actively probe the actual target path and environment (`test_prediction.py:64`, `:151`). No heldout action is supplied to construction; the frozen query receives DOWN only at prediction time. No heldout assimilation or constructor feedback follows scoring. These are local boundary results, not claims about a future provider adapter.
- **Prediction before scoring:** `predict_once` has no target-path argument. It replays the known prefix, makes one separate heldout predict RPC, then writes raw output, validated commitment and a hash-bound receipt (`pipeline.py:125–146`). `score_bound` checks the externally retained binding digest and all input/output hashes, consumes scoring, and only then opens the target (`pipeline.py:151–174`). The retained demo events and printed binding match this order; ordering tests guard the actual file read (`test_prediction.py:87`, `:114`). Trusted preparation and this reviewer do read historical outcome bytes; they are not the sealed candidate or the prediction phase.
- **Consumption/tampering:** exclusive creation with fsync (`evidence.py:9–15`) plus the shared source/prefix/query registry (`pipeline.py:120–124`) blocks reissuing the same query under another output directory. Consumption precedes execution and remains after failure. Source, prefix, query, raw prediction, validated prediction and BOUND mutation are tested before target access. Scoring is terminal after a malformed target. This is workflow integrity under a trusted host, not cryptographic protection from a host operator who deletes controls, supplies a different registry or rewrites the complete evidence chain.
- **Nonvacuous metrics:** exact match, coverage, abstention, cell accuracy, identity baseline, signed differences, changed-cell accuracy and false changes are independently distinguishable (`prediction_contract.py:28–37`). Abstention is coverage 0 with null prediction metrics and no predictive success. Undefined denominators remain null. Correct unchanged identity is a valid exact answer with zero improvement. Tests cover identity, exact fixtures, abstention, malformed targets and undefined denominators.

## Actual recording and independent demo arithmetic

The preserved trace hashes to `88ae05507b1a2c23e169bcb2c320d74e31f739ef5096004bb988e540dbc26f80`. Its first two rows are step 1/turn 0 RIGHT and step 2/turn 1 DOWN, continuous frames, level 0, NOT_FINISHED. Prefix bytes and the selected target independently match those rows. Session `cd82-fb555c5d-b7azqbdf` is bound by the previously retained numeric-version collection mapping, whose hash is `5a7a4634addf4bb089c9eee9fe7406b8aa91cdcd41862737c84b95547bcee317`.

This provenance refers to the actual returned `pranavagrawai/r10a-cd82-constructor-repair`, version 1. It does not relabel that run as the intended R10 target or waive its mismatch. No new provider lookup was made.

The demo's 41 preserved files match their hashes. Its externally printed BOUND hash is `9c673b9af6bb74ff11c0af5b6990b946514725f2384fec77d8019678258d3dcd`. Independent arithmetic without importing the candidate scorer gives:

| Measure | Identity fixture |
|---|---:|
| Cells / changed / unchanged | 4096 / 200 / 3896 |
| Coverage | 1 |
| Exact match | false |
| Cell accuracy | 0.951171875 |
| Changed cells correct | 0 / 200 |
| False changes | 0 |
| Improvement over identity | 0 |

These values match SCORE.json. This is a manufactured identity source on a real recorded target, not a model-generated prediction. No competence, dynamics efficacy, treatment effect or R10b readiness follows.

## Scope and provenance limits

1. The offline `freeze-artifact` path rechecks supplied exact source under the typed contract, preserves it and rejects incompatible source; it does not generate, convert or repair a model. Its integration test uses a supplied fixture artifact. No new live model result exists in this package, and old ABI-only freezes are not automatically compatible.
2. Historical operators have already seen the recording. The fixed first/second-row selection is reproducible and the card is hash-bound. File hashes do not independently prove the historical time at which an operator first examined the target. Treat this as a retrospective chronological holdout isolated from the current sealed candidate, never a globally untouched prospective dataset.
3. Fixture exact answers are deliberately manufactured evaluator tests. Local sandbox tests establish the exercised boundaries, not exclusion of every conceivable side channel. The host and preparation operator remain trusted.
4. A future paid adapter needs its own exact-byte review of transport, budgets, treatment, provenance and release controls. This PASS supplies no paid authorization or launch approval.

## Preservation and reproduction

Independently verified **3479 prior OpenRouter files**, **70 R10 frozen source bindings**, all 42 current inventory members and all 41 demo bindings. R10 manifest remains `9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9`. The read-only candidate checker also passes.

From repository root:

```bash
python3 -B analysis/constructor-prediction-test-20260916T233048Z/independent-review/audit.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/constructor-prediction-test-20260916T233048Z/test_prediction.py --basetemp=analysis/constructor-prediction-test-20260916T233048Z/independent-review/NEW-UNUSED-BASETEMP
```

Use a fresh basetemp. Do not repeat the consumed recorded demo. `audit.py` reads preserved bytes and recomputes hashes/arithmetic without executing candidate code or scoring a new attempt. An initial reviewer assertion incorrectly treated the CLI's JSON stdout as a bare hash; the audit now parses the actual documented JSON output. This was a reviewer script correction, not a candidate defect.

Permitted next decision: accept this offline package as mechanically ready for a separately supplied compatible artifact, or separately scope preparation of a paid entry. No model call is authorized by this report.
