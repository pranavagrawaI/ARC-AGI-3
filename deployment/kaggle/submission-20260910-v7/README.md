# Public endurance candidate — 10 September

Successor to the preserved, public-qualified v5 candidate. This notebook explicitly
refuses competition rerun mode. No competition submission is authorized.

Runtime: submission-runtime-20260910-v4. See BUILD.json for exact identities.

## Limits and deadline hierarchy

- Three public games: LS20, S5I5, CD82, sequentially.
- Model-call admission: 600 seconds per game; maximum 200 executed Solver actions.
- Same Qwen3.8-27B-FP8 route; routine reasoning off; no background summarizer.
- Work deadline: 3100 seconds from the first notebook code cell.
- Outer process hard deadline: work deadline + 200 seconds, leaving 300 seconds
  before a 3600-second Kaggle session cap.
- Server readiness timeout: 1200 seconds (required by r8-l3b). Launcher timeout: at most 2700 seconds,
  reduced after startup to leave 30 seconds before the work deadline.
- The run refuses to start a game if remaining time cannot fund its full
  precommitted 600-second admission window. It does not silently shorten the test.
- The work alarm enters the outer runtime's exception/finally cleanup path.
  That alarm is disarmed before cleanup. A wedged cleanup remains a failure;
  the independent hard watchdog is a containment fallback, not a clean PASS.
- Per admitted request: client timeout 90 seconds, retries zero. Per-game request
  and tool reserves remain 90+15 seconds; cleanup reserve remains 30 seconds.

Bootstrap exhaustion, request failures, stuck tools, or infrastructure errors do
not become normal budget stops. The first model request exception now writes a
failure marker, blocks further admission, and fails the host session promptly.
Cleanup failures are attached to the original error rather than replacing it.

## Measurement

Existing request lifecycle records now include peak worker RSS in KiB (Linux
ru_maxrss), and completed-call records carry request IDs. execution_timing.jsonl
timestamps each host-received action result and its level count. These permit
first/second five-minute comparisons tied to admission_deadline - 600. They do not
measure exact GPU peak memory or isolate server-only token generation time.

The model sees remaining budget in its prompt, so the 600-second run is not a pure
time extension of the 75-second treatment. It is an endurance qualification of
this explicitly configured candidate, not a causal score comparison.

## Local checks

    deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/submission-20260910-v6/test_submission.py deployment/kaggle/submission-20260910-v6/test_budget_accounting.py deployment/kaggle/submission-20260910-v6/test_deadlines.py -q

Tests cover normal/exhausted gameplay and delayed HTTP, HTTP 503 and stalled
Python tools through offline and local HTTP-gateway entrypoints, request failure
admission blocking, process retirement, real outer work-alarm cleanup, timing
records, payload identity, budget arithmetic, and refusal of competition mode.
These are controlled local checks, not target-GPU fault-injection evidence.

Build: deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v6/build.py.
