# Epistemic contract endurance result

Implemented the agreed contract and ran one same-format public Kaggle endurance
qualification. Operational result: PASS. Solving result: zero completed levels.
The run does not establish that epistemic management improved.

| Game | Executed actions | Model calls | Levels completed | Stop |
| --- | ---: | ---: | ---: | --- |
| LS20 | 9 | 53 | 0 | Model admission budget |
| S5I5 | 24 | 65 | 0 | Wall budget |
| CD82 | 19 | 59 | 0 | Wall budget |
| Total | 52 | 177 | 0 | |

Game-session time was 1808.324 seconds (30m08s). Actor requests occupied
1769.092 seconds, 97.83% of that time. Notebook final log time was 2201.009
seconds (36m41s). All three received their full 600-second admission allowance.
177 solver requests plus one warmup reconcile with 178 successful server
accesses. Source hashes, embedded runtime, remote source readback, per-game
request ledgers, budget stops and TERM-only server shutdown passed audit.
No model retries, reasoning tokens, summarizer calls or operational failures
were observed by the audit. Monitoring-stream reconnects were separate from
model requests; persisted logs were retrieved afterward.

## What the new contract actually did

LS20 made two recorded planner calls at step 7. Searching for bar_left=54
with depth 10 returned limited/depth; the Actor recognized the remaining
34-step distance and chose another target. A local band-position target then
returned UP, RIGHT, with fresh replay matched. The two searches together took
0.02547 seconds and their children were reaped. This demonstrates live
reachability and correct interpretation of that cutoff, not useful execution
of the returned plan.

There were zero search_id-linked submissions in all games. The Actor instead
constructed a 14-UP batch manually for an unvalidated bar-depletion question.
The host executed two actions, detected a frozen-model mismatch at step 9,
and discarded the remaining 12. The model had assumed a narrow movement
window and its encoder searched only that crop; the actual block moved above
it. The Actor subsequently revised that assumption. The live search-to-submit
provenance join therefore remains unexercised by the Actor, although local
real-agent tests exercised both rejection and successful submission.

## The epistemic failure persists

S5I5 provides the clearest evidence. Its notes initially name alternatives,
but after three probes state: "Confirmed: EVERY click (any cell) increments
the bottom-row counter" and "Object bodies never change." The notes then
adopt a "Pure click-count game" objective hypothesis and pursue the counter.
22 of 24 actions were CLICK 0 0; all 24 transitions changed only row 63.
Later effort refined the counter's increment pattern and repeatedly reopened
batching. It did not establish the counter's relation to level completion.
This does not prove the objective hypothesis false; it demonstrates that a
small tested scope became a universal rule and an unvalidated policy.

LS20 initially used the requested claim/evidence/scope/open-conditions table,
but later used a matching prediction to reinforce a guessed movement boundary.
A correct diagnostic result and the presence of uncertainty words did not keep
the next commitment appropriately scoped. CD82 used more varied actions and
nine transitions changed cells outside row 63, but completed no level. Its
notes' interpretation of black-cell changes as collection remains an Actor
hypothesis, not a host-certified objective.

These are specific trace observations, not a scored claim about overall model
judgment. There was no paired control: repairs, prompts, evidence labels and
planner surface changed together. The historical run had 153 calls and 68
executed actions with zero levels; the new run had 177 calls and 52 actions.
That descriptive difference does not isolate a treatment effect.

## Delivery and qualification

Candidate: deployment/kaggle/submission-20260910-v9, containing
submission-runtime-20260910-v6. Runtime contract:
061369c045635581ac5e4d9fea99d532722f3a3e5d0c4ce93b4b75475a7876ea.
Kaggle: pranavagrawai/arc3-first-submission-20260910, kernel version 7.
Local qualification: 99 runtime tests + 35 notebook/real-entrypoint tests +
4 contingent-worker tests = 138 passed. Exact notebook command/payload and
source freeze were checked. Repaired v5 and historical v4 stayed unchanged.
The earlier local suite invalidated by overlapping final edits is preserved
under pre-freeze-checks/; the frozen final suite passed.

The implementation includes optional exploration without a fabricated goal or
hypothesis pair, scoped evidence/retention guidance, measured-prediction-only
experiment authority, hard-bounded model search, replay, distinct failure
statuses, shared model selection and state/helper-bound plan references.
Semantic uncertainty management remains a reasoning requirement rather than
a property certified by these structural checks. This run demonstrates that
the instructions and labels alone did not enforce the intended behavior.

Retain the operational qualification and these observations. Do not promote
this as a solving or epistemic improvement. No automatic rerun or competition
submission was made. The next design question is the observed promotion of
three probes into a universal rule and a spending policy, not planner speed.

## Reproduce the measurements

- `python3 analysis/deployment-probes/20260910-epistemic-endurance-v1/audit.py`
- `python3 analysis/deployment-probes/20260910-epistemic-endurance-v1/inspect_behavior.py`
- Local verifier: `deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v9/verify_repairs.py`

Evidence: [audit](AUDIT.json), [behavior measurements](BEHAVIOR.json),
[batch interruptions](BATCH-STOPS.json), [implementation](IMPLEMENTATION.md),
[immutable card](EXPERIMENT-CARD.md), [local qualification](VERIFIED.json).
Raw source, helper versions, transitions, model-generated code and notes are
under raw/qualification/solver-workspaces/. S5I5's critical scope promotion is
in s5i5-18d95033-4dz_el0e/team_nemo/shared/knowledge/step_log.md, lines 52–67.
LS20's corresponding boundary inference and correction are in
ls20-9607627b-4ztvex5y/team_nemo/shared/knowledge/step_log.md, lines 152–167.
