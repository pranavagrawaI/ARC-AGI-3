# 0158 LS20 long-run attempt v19

## Result

The fresh run recorded successors for **six actions** before an ARC service
read timeout at authorized action 7 made execution status uncertain. The run is
terminal and cannot be resumed or replay action 7.

- Run: `artifacts/live-ls20/20260812T141806Z-qwen36-autonomous-44`
- Run ID: `d47924f2b2124ccc9956d37c67916ee8`
- Confirmed actions: 6 (`1, 2, 3, 4, 1, 2`)
- Authorized but unconfirmed: action 7 (`3`)
- Levels completed: 0 confirmed

## Behavioral evidence before substrate failure

The first two predictions were contradicted. Qwen then produced three linked
exact replacement predictions that the verifier supported, reaching G2 at
action 5. The host recorded G2 without exposing a new API and autonomously
reset the grounding-only cycle. Action 6 also completed with a successor.
This demonstrates revision from probe history and sustained use of the
prospective protocol beyond the original one-action boundary.

## Failure and host defect

The ARC SDK hard-coded a 10-second POST timeout. Its action-7 request timed out
without returning a public successor, so the host could not determine whether
the service applied it. Retrying would risk a duplicate action and was
correctly forbidden.

A second defect prevented normal terminalization: the external-failure packet
passed `environment_step` to a formatter that accepts only claim or commitment
phases. `posthoc-terminal-recovery.json` records six confirmed successors, one
unconfirmed authorized commitment, and `resumable: false` without pretending a
normal scorecard existed.

The next candidate fixes packet formatting, explicitly terminalizes pending
uncertain actions, and raises the SDK read timeout to 60 seconds without adding
POST retries. The executed 20-file candidate was hash-checked and snapshotted.
