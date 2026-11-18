# CONSTRUCTOR-PREDICTION-TEST — 20260916T233048Z

Status: local preparation only; NO paid calls, credentials, remote run, Kaggle or new gameplay authorized.

Question: Can a frozen executable constructor candidate make a concrete complete next-grid prediction before the next observation is supplied, and can the scorer separate coverage, accuracy and improvement over copying the current grid?
Posture: mechanical prediction test preparation. Model evidence absent. Fixture results are explicitly synthetic, not model competence.
Contrast: frozen prediction versus unchanged-current-grid identity baseline on exactly the same recorded action/target. Future reasoning off/on would share this interface, but is NOT AUTHORIZED.

## Selection fixed before holdout inspection

Select the existing R10 B recording at `analysis/r10-prep-20260916/exact-collection-20260916/raw/model-runtime/solver-workspaces/arm-B/cd82-fb555c5d-b7azqbdf/ipc/transitions.jsonl`. Construction prefix is exactly its chronological step 1. Holdout is exactly the immediately next recorded step 2, with its actual action. Require contiguous before/after observations, same recording session, no reset or level/episode boundary. If invalid or absent, BLOCKED; do not select another case based on changes or accuracy. Never choose a target after seeing whether the candidate is correct.

This is a RETROSPECTIVE CHRONOLOGICAL HOLDOUT, model-unseen in this invocation, not a globally untouched prospective dataset. Operators and historical runs have seen these data. A single next step is a mechanical pilot, not a competence test.

## Prediction and protocol

Prediction: identity fixtures will have zero improvement over the identity baseline, even when full-grid accuracy is high; exact manufactured predictors score exactly correct; abstention yields zero prediction coverage and no predictive success. On a real recorded case no model-performance prediction is asserted before a model candidate exists.

Preserve initialize(observation)->dict, predict(state,action)->one branch, assimilate(predicted_state,observation,verdict)->dict. The branch commitments must be EXACTLY one of:
- {"type":"grid","predicted_grid": [[integer, ...], ...]} with complete dimensions equal to current observation.
- {"type":"abstain","reason":"nonblank explanation"}.
No empty/prose-only/placeholder commitments, mixed alternatives, nonfinite/noninteger cells, missing rows/cells or dimension mismatch. State/commitment roles remain separate. Candidate chooses its own hypothesis; host supplies no transition rule. Abstention is valid coverage reporting, never predictive success.

Constructor gets only step-1 prefix. Next action is supplied only at frozen prediction time. Source freezes before query. Replay the visible prefix to build state, using only its observations and unscored assimilation; then exactly one heldout predict call. Persist source-bound immutable prediction/result digest before scoring process reads heldout outcome. No target feedback, assimilation on holdout, retry, candidate repair or new construction after target reveal. Prior files remain untouched.

## Measures (fixed definitions)

Let C=current grid, Y=heldout target, P=concrete prediction; N=number of cells, D={i:Y_i!=C_i}, U={i:Y_i=C_i}.
- Executable construction: visible ABI/typed-commitment replay passes; separate from prediction correctness.
- Coverage: 1 for valid concrete grid; 0 for explicit abstention. Malformed prediction is invalid, separately counted.
- Exact full-grid match: P==Y. Full-grid cell accuracy: count(P_i==Y_i)/N.
- Baseline: P_base=C under the identical next action and target; report its exact match and cell accuracy.
- Paired difference: prediction cell accuracy minus baseline cell accuracy; also signed exact-match difference. One-case descriptive differences, no inference/statistical claim.
- Changed-cell count: |D|; changed-cell accuracy: count(i in D:P_i==Y_i)/|D|; null/undefined when |D|=0. Baseline changed-cell accuracy is 0 when |D|>0, otherwise undefined.
- False changes: count(i in U:P_i!=C_i); rate divided by |U|, undefined when |U|=0.
- Abstain: prediction accuracy/exact/paired differences and prediction diagnostics null, not 0 or a success; baseline metrics remain available. Invalid output: no candidate score, retain failure.
High accuracy dominated by unchanged cells is not superiority. Exact identity on truly unchanged data is legitimate correctness but zero improvement. Do not require any movement or semantic rule.

## Numeric caps / stops

Current task: $0 paid budget, 0 model/remote/credential calls, 0 gameplay actions. Local fixtures only; one holdout query per frozen-source/output identity; no retry after consumption/failure. Reuse constructor 6 requests, 10 inspections, 4 submissions, 4096 requested completion/request including reasoning, 24576 completion and600000 prompt/episode, 240s soft/245s hard; in this task all responses are loopback fixtures. Candidate sealed RPC limits remain inherited (15s replay boundary, <=3s/call; separate heldout query boundary); no containment fallback. Raw source/JSON limits inherited.
A future paid off/on pair could retain $2 total admission, same rates/provider/reasoning treatment, but needs a new exact card and independent/PM authority; no paid entry is authorized here.

Stop on invalid provenance/boundary, source/prefix/prediction drift, pre-prediction target access, malformed prediction, containment failure, timeout, already consumed output, or any request to repair/retry after holdout. Preserve failed receipts.
Invalid if holdout/next action enters construction, outcome read before prediction binding, source/prediction modified after binding, target feedback/assimilation occurs, case selected based on scoring outcome, scoring accepts wrong shape/type, unscored ABI freeze is called predictive support, or fixture/model evidence is conflated.

## Result map / outputs

Exact valid predictor above baseline -> only descriptive evidence on this one recorded transition, separately authorized broader study required. Equal baseline -> no demonstrated improvement (even if exact unchanged result). Worse -> descriptive failure, no retry. Abstain -> missing coverage, no predictive success. Invalid/leakage/drift/boundary -> HOLD; do not score or repair to success.
Outputs: new-directory card, source/trace provenance, separated prefix/query/holdout files, constructor transcripts, immutable freeze/query/prediction/scoring receipts, hashes, tests, README commands and review inventory. No model result claimed in preparation. Reviewer owns independent approval; no self-approval.
