# Observation/action consistency pilot: blocked before comparison

Classification: INTEGRATION_INVALID. The first generation request, seed 51 baseline turn 0, received HTTP 429 from DekaLLM's upstream shared pool. The provider reported that qwen/qwen3.8-27b was temporarily rate-limited. The pilot stopped immediately with no retry, fallback, alternate provider, or second launch.

## Live outcome

- Generation attempts: 1. Completed model responses: 0.
- Completed arms: 0. Model-proposed or newly executed game actions: 0.
- The offline environment prefix replay completed before the attempted generation; those seven historical actions are setup, not new agent progress.
- Request latency: 2.203439 seconds. Overall launched pilot time: 3.069213 seconds.
- Conservative failed-request reservation: $0.023581 under the $1 total cap.
- Actual billed tokens and cost: unknown because no usage receipt was returned. `receipted_cost: 0` is an empty sum, not evidence of zero spend; `total_cost` is null.
- The authenticated prelaunch credit check passed. No further provider request, including a post-run credit check, was made after the generation failure.

This is an infrastructure failure, not evidence for or against consistency enforcement. No baseline-versus-treatment result exists.

## Addition implemented and exercised locally

The new runner imports the earlier frozen replay and transport code read-only. Each decision contains a model-selected color readback with its count and inclusive current-frame bounding box, plus separately declared and submitted action sequences.

Both arms receive the same exact generic per-color geometry table, history, notes, budgets and decision schema. The baseline records the validation in shadow mode without blocking or revealing findings. The treatment adds the consistency instruction, blocks false readback or mismatching action sequences before any action, and returns only factual feedback. A repair consumes the existing four-call budget; the host does not infer mechanics or select a move. The earlier assumption-to-probe checkpoint is absent.

Seven new offline tests and nine frozen baseline tests passed. The new tests executed actual local environment actions with explicitly synthetic provider fixtures. They verified a rejected mismatching proposal followed by a corrected executable submission, baseline non-enforcement, repeated rejection without side effects, response-only metadata removal on the next request, numeric measurement correctness, schema rejection, missing usage, and first-request HTTP failure accounting. These tests establish machinery behavior, not model improvement. Both suites emitted a cleanup ResourceWarning for synthetic HTTPError objects; the tests passed.

Preparation verified the complete seven-action prefix twice against archived frames, state and levels; checked the initial request size; inspected live DekaLLM endpoint parameter support; and froze source, experiment card, endpoint metadata and test evidence. The audit confirmed the source and prelaunch hashes remain unchanged.

## Audit interpretation

`AUDIT.json` passes the narrow accounting checks and marks performance invalid. `FINAL-AUDIT.json` has `passed: false` because its required `complete` check is false. Its source, routing, accounting, wall-cap and cost-reconciliation checks pass. The failure was preserved instead of changing the frozen runner to force a passing overall verdict.

## Decision and limits

Keep the incumbent unchanged. A future attempt requires a new experiment identity and authorization; do not rerun this frozen launch. The precommitted three paired continuations remain untested live.

Even a successful successor would measure enforcement against a newly instrumented baseline sharing the exact geometry table and extended schema. It would not isolate the value of the geometry table relative to the previous pilot, prove natural-language rationale/action agreement, or establish full-solver performance. A correct readback of one color and equal action lists are deliberately narrow contracts.

## Evidence

- `EXPERIMENT-CARD.md`, `PRELAUNCH.json`, `STARTED.json`, `ENDPOINTS.json`
- `TESTS.txt`, `BASELINE-TESTS.txt`
- `raw/00-request.json`, `raw/00-error.body`, `attempts.jsonl`
- `RESULT.json`, `AUDIT.json`, `FINAL-AUDIT.json`
- `run.py`, `test_run.py`, `audit.py`

Offline audit command:

    uv run python analysis/observation-action-pilot-20260913/audit.py

It intentionally exits nonzero for this incomplete run. No old experiment or deployment files were edited and no Kaggle or competition submission occurred.
