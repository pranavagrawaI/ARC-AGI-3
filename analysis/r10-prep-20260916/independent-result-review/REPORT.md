# R10A result review — DONE

## Decision

**The bounded descriptive constructor diagnosis is supported. Remote freeze/use acceptance was not met; existing target-mismatch HOLD remains intact. No efficacy claim or R10b clearance follows.**

Independent offline checks reproduce the significant raw facts in `../exact-collection-20260916/RESULT.md` and `RECONCILIATION.json`. The observed failure is a source-representation failure at ABI initialization, before any model prediction can be evaluated. It is not evidence that the candidate's game dynamics were tested and found wrong. No frozen code or original report was changed.

## Delivered contract versus returned representation

All paths below are relative to `../exact-collection-20260916/raw/model-runtime/solver-workspaces/` unless stated otherwise. JSON locations are supplied because wire files contain long single-line records.

- Actual outbound requests `raw/raw-requests/074.wire.json` through `079.wire.json`, `/messages/0/content`, require complete Python defining initialize/predict/assimilate and explicitly distinguish ABI replay from semantic validation. The user packet's `required_submission.candidate_source` calls for complete Python source. Calls 77–79 offer `submit_candidate`; `/tools/*/function/parameters/properties/candidate_source` accepts **a string of source text or an actual JSON array of source-line strings**. Its description prefers source lines and warns against double escaping.
- There is a real wording inconsistency in the system message: one sentence says candidate_source “must be complete nonempty text,” while later instructions and the schema explicitly permit a list. The frozen implementation supports both. This merits recording as prompt ambiguity, but **neither interpretation licenses a string whose Python content is a list expression**. The evidence cannot establish whether that inconsistency caused the model's choice; it does not show host transport converted a conforming array into a string.
- Raw response bodies `raw/raw-http/77.body`, `78.body`, `79.body`, `/choices/0/message/tool_calls/0/function/arguments`, decode to candidate_source of JSON type **string**. Static Python parsing of each produces exactly `Expr(List(Constant strings))`, with no function definitions. Source hashes are respectively `99b94bd6f18bdd5fb4c2d1d56011ea3ae9a1dc678c6d238d12ae5e177b32d9b5`, `eaecc82191c18090db5fbd8e9a3e634250ddbd53c86ced31b84c885d26ebee61`, and `5bca6e08c9d152a49f07f183cd54f5fdaf52f5f2dbef8d12fc71373071482d11`. All match the evaluator's source hashes.
- The raw tool calls equal `arm-B/cd82-fb555c5d-b7azqbdf/tool-events.jsonl:4–6`; parent and child event files are byte-identical. Frozen `../r10a-package/constructor.py:326–338` joins an actual list into source and passes a string unchanged after compilation. Compilation succeeds for a list expression but defines no functions. `../r10a-package/integration_boundary.py:16` appends the trusted adapter **after** the source. Thus the last submission's claim that the wrapper binds names before source execution is contradicted by the source and traceback; wrapper line 4 reflects the single-line list expression plus appended adapter.
- Every failure reports `NameError: name 'initialize' is not defined`. Calls 78 and 79 contain the exact preceding tool messages and IDs, including the NameErrors. All six outbound requests carry exactly the preceding constructor feedback sequence. Feedback for the final submission is recorded locally; there is no seventh model request proving the model saw that final result.

**Attribution:** the local model's returned tool arguments contain the incorrect representation before host normalization. This establishes the proximate failure without proving a general model incapability, an optimal prompt, or a semantic defect in the unexecuted proposed model. No candidate code was executed or transformed into a repaired candidate by this review.

## Execution, publication and scoring

Three inspection outputs have return code 0. The third inspection's full reported text matches a fresh raw-grid difference calculation: B step 1, RIGHT, 64×64 grids, 201 changed cells, matching first 80 tuples. This establishes inspection access/output for this prefix, not the correctness of the model's interpretation of color 15 as an actor.

Parent and child `RESULT.json` agree: BUDGET_EXHAUSTED, inspections 3, submissions 3, model_requests 6, frozen false, prospective not_run, supported_model false. Six constructor calls are precisely relay calls 74–79. Their raw usage sums to 127,911 prompt and 2,844 completion tokens; maximum HTTP duration is below 30 seconds. Parent construction records report the approximately 84.139-second episode, below the wall limits; it stopped after the six-request ceiling, not after a successful freeze. Three failed submissions with no accepted repair yield the descriptive episode repair fraction 0/1; this is one episode, not an estimated general repair rate.

No authoritative root freeze or child freeze is supported by the result/launcher records; exact freeze-path 404s corroborate only those named-path observations. B has no completed frozen prediction or assimilation events. A's host-events file was not collected as a successful artifact; its absence is not independently treated as proof. Together with the launcher, role ledger, absent A construction calls and retained unavailable prediction records, the evidence supports no recorded A construction/use. No prospective action/state binding, null-state handling, semantic scoring, or final-pending assimilation was exercised remotely.

Independent comparison finds A 10 and B 2 IPC transitions and matching official successor frames, with corresponding IPC before/after grids and official state/level values. Both finish NOT_FINISHED with zero levels. The recording action IDs are all zero; **action names remain IPC testimony**, not independently corroborated by those IDs. Incumbent prediction and feedback statuses are all unavailable (10 A, 2 B). The advertised ABI checker does not score commitments; here it did not even complete initialization. An empty commitment set or an ABI pass would not itself be semantic support. These observations support neither treatment efficacy nor verified frozen-model accuracy.

## Accounting, teardown and missing request 024

The audit verifies 330 distinct requested paths and 316 retrieved-file byte hashes against individual numeric-version receipts; no duplicate requested path appears. All 105 response bodies match their response ledger and launcher hashes. Independent total usage is 4,197,430 prompt and 37,288 completion tokens. The 104 present raw requests match their declared message digests. Server log has 106 successful completion accesses, consistent with 105 application calls plus the separately recorded warmup. These are runtime accounting records, not an independent billing measurement.

Raw lifecycle evidence has five owned processes and two GPU processes before shutdown, zero after, with no remaining/unresolved listener inodes. Seven closed scans remain closed over about 5.018 seconds, beginning at 5.754 seconds after TERM. Shutdown, bootstrap reap and process-guard receipts corroborate closure; these checks do not certify provider billing or future physical state.

`raw/raw-requests/024.wire.json` remains absent after its preserved HTTP 429. Relay call 24 is **Arm A actor**, while the constructor is B calls 74–79, all with raw requests and responses present. The exact-version A IPC message sequence, projected through the frozen `local_wire.project_messages`, independently hashes to `d33bf96b4506da4f33905465a652b8713cdb2e027c4ae5fedd952a952f0d8cdd`, matching the relay. This corroborates the messages only; it does not reconstruct or replace the missing complete wire request. The gap prevents declaring a complete raw-request archive, but does not remove any link in this constructor prompt/schema/response/feedback diagnosis. No retry was made.

## Reproduction, bindings and permitted next decision

Run from the repository root:

```bash
python3 -B analysis/r10-prep-20260916/independent-result-review/audit.py
```

This is offline and writes only this review's `CHECKS.json` and `INPUT-HASHES.json`. `check-output.json` retains the successful run summary; the script binds every read input and checks stability before publishing. An initial reviewer-script run incorrectly assumed A had a host-events file and stopped with FileNotFoundError; the script now explicitly handles that known absence, with the evidentiary limit stated above. No input was changed to make checks pass.

`INPUT-HASHES.json` preserves exact bindings for the raw collection, original reports/controls, inspected frozen source and receipts. `EVIDENCE-HASHES.json` binds the review artifacts. Changed inputs require renewed assessment of affected conclusions.

**Permitted next decision:** PM/Hermes may use these preserved bytes to decide whether separately authorized future work should address the representation failure and prompt wording. This review implements no repair, proposes no launch, waives no identity mismatch, and does not clear R10b. The failed remote freeze/use criterion and incomplete raw archive remain explicit; original reports and controls are preserved.
