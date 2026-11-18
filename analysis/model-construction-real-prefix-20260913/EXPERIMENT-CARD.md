# LS20 real-prefix construction integration episode

Status: authorized prelaunch card; immutable once STARTED.json exists

## Claim under test

Given only the chronological LS20 prefix through transition 5, the incumbent executable model, the already-proposed fixed action sequence, and host-owned factual diagnostics, the specified OpenRouter model can select a useful model-building work item and produce an executable revision that:

1. preserves informative predictions on all five visible transitions;
2. exposes its operation, dependencies, remaining scope, and completion checks;
3. is frozen before suffix access; and
4. preserves a jointly compatible candidate path farther into the sealed action-conditioned suffix than the incumbent.

This is one integration episode. It cannot establish game-general model-construction competence or a C-over-B treatment effect.

## Objective

Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.

## Case and split

- Source: preserved LS20 sparse-reasoning public run, workspace `ls20-9607627b-jz1v9k2m`.
- Visible frontier: transitions 1-5 inclusive.
- Decision-time proposed actions: `UP, UP, LEFT, LEFT, LEFT`, committed at turn 5 before any successor in that batch was observed.
- Sealed suffix: executed transitions 6-8 (`UP, UP, LEFT`).
- Incumbent: exact source committed at turn 5, SHA-256 `fbd0a8187e45fe3bbf05567ce3e25052f2f00f4f9277429af95e3c81a8f78816`.

This is historical evidence already known to experiment designers, withheld only from this constructor. It is not an untouched scientific holdout. The implementation is a narrow independent integration with singleton observable values per branch, not the full reference prototype (no finite-domain informativeness calculus or static dependency-completeness proof).

The constructor must not receive suffix observations, suffix assessments, or a host diagnosis such as “collision” or “terrain canonicalization.”

## Host authority

The host may establish chronology, freeze bytes, execute candidate code in an isolated realm, extract evaluator-owned observable targets (`block`, `bar_left`), assess branch-local conjunctions, preserve ancestry, enforce budgets, and report factual mismatches.

The host must not identify the semantic cause of a mismatch, supply an applicability predicate, choose the constructor's work item, or treat newly assimilated information as retroactive prediction credit.

Declared dependencies, dependencies exercised on visible cases, and dependencies encountered on the proposed rollout are reported separately.

## Construction policy arm

Operation types: `ground`, `contrast`, `close`, `compose`, `test`.

Lifecycle: `OPEN -> WORKING -> CANDIDATE_READY | NEEDS_EVIDENCE | BUDGET_EXHAUSTED`.

The model selects the work item. A work item states the blocked decision, obstacle hypothesis, operation, claim, evidence references, dependencies, completion checks, and remaining scope. Operations may repeat or be skipped.

Sparse investigation distinguishes:

- needs computation;
- needs observation;
- does not block this decision.

Reopening an obstacle requires new evidence, a distinct candidate, or explicit additional search budget.

## Execution semantics

Candidate ABI calls run in fresh processes with JSON state as the only persistence. The subprocess receives only source, an ABI-only runner, and current call arguments; no observation bundle or trusted evaluator is mounted. Observers, joint assessment, ancestry, and byte identity remain host-owned.

For every visible transition:

1. predict from the current version-bound belief;
2. freeze each branch's joint observable commitments;
3. measure the successor with the frozen host observer;
4. assess commitments;
5. only then assimilate the observation into surviving predicted branches.

After a model revision, reconstruct the current belief by replaying the visible prefix under the new frozen version.

The sealed fixed-action rollout is built completely before suffix observations are read. Suffix audit prunes paths chronologically; a later observation cannot resurrect an earlier rejected path. No suffix observation repairs simulated physical state.

## Budgets

- Model: `qwen/qwen3.8-27b` through OpenRouter.
- Routine reasoning: disabled.
- Maximum model requests: 6 total, including tool continuations and candidate revision.
- Maximum completion tokens: 8,192 total requested across calls.
- Maximum Python inspection calls: 4.
- Maximum candidate submissions: 2.
- Per-request timeout: 90 seconds.
- Total wall limit: 600 seconds.
- Branch cap: 32; overflow is reported as resource exhaustion, never evidential elimination.
- No automatic retry. A new attempt requires a new experiment identity.

## Predeclared outcomes

- `INTEGRATION_VALID_CANDIDATE_SUPPORTED`: the full path executes, the constructor autonomously submits a candidate that passes visible checks, and its one-time suffix audit improves over the incumbent.
- `INTEGRATION_VALID_CANDIDATE_NOT_SUPPORTED`: a candidate passes visible construction checks but does not improve sealed-suffix support.
- `INTEGRATION_VALID_NEEDS_EVIDENCE`: the model returns an explicit evidence request without a qualifying candidate.
- `INTEGRATION_VALID_BUDGET_EXHAUSTED`: bounded construction terminates without a qualifying candidate.
- `INTEGRATION_INVALID`: chronology leakage, suffix access before freeze, sandbox failure, malformed evidence ownership, transport failure, or budget-accounting failure invalidates the episode.

A visible-prefix pass is construction completion, not prospective support. Suffix feedback, once exposed, becomes evidence for a future experiment and cannot be reused as sealed evaluation.

## Verification before launch

- Reference prototype SHA-256 manifest verifies.
- Its 28 synthetic tests pass locally.
- Real-trace split and seal tests pass.
- Incumbent passes the visible prefix and fails at the third sealed action.
- A host-only witness candidate proves that the evaluator can accept a better trajectory without being exposed to the model.
- Generated code executes only inside bubblewrap with no network, a read-only evidence mount, bounded time, and bounded output.
- Live execution refuses to start without `OPENROUTER_API_KEY` already present in the process environment.

## Precommitment result map and artifacts

Question: can this bounded constructor produce a locally better executable trajectory model?
Posture: offline integration feasibility and diagnosis.
Contrast: historical incumbent on the same suffix; no randomized treatment comparison.
Prediction: candidate may pass visible checks, but improved historical-suffix support is uncertain.
Measure: `RESULT.json.classification`, `raw/construction.json`, `raw/FROZEN.json`, and `raw/sealed_suffix_evaluation.json.compatible_depth` against incumbent depth 2.
Spend: at most six requests and 8,192 requested completion tokens; prompt tokens billed additionally. No independent USD cap was authorized or inferred; these exact request/token caps are the authorized spend boundary.
Stop: first qualifying candidate or valid evidence request, exhaustion of any episode cap, endpoint/transport failure, or 600 seconds. No transport retry or model substitution.
Invalid if: the invalidity conditions above, reasoning used despite disabled request, or evaluator failure.
Result map: supported → preserve candidate for a separately authorized new-evidence experiment, without deployment; not supported → retain machinery and inspect model limitations; needs evidence → preserve the request without supplying new observations; budget exhausted → preserve bounded negative construction evidence; invalid → stop, diagnose the integration/endpoint, and do not rerun this identity.
Outputs: exact request JSON plus raw response bodies under `raw/requests/` and `raw/responses/`; durable attempts `raw/request_attempts.jsonl`; calls/results `raw/tool_events.jsonl`; final `RESULT.json` and `RESULT.md`.
