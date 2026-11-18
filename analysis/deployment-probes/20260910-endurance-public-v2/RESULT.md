# Public endurance v2 — operational PASS, no levels completed

Kaggle version 6 completed and passed the frozen card's operational checks.
This qualifies the exact candidate for these three public ten-minute admission
windows; it does not establish reliable solving, comparative benefit, nine-hour
endurance, or competition readiness. No competition submission was made.

## Recorded outcomes

| Public game | Session seconds | Completed Actor calls | Executed actions | Levels completed | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| LS20 | 618.24 | 51 | 6 | 0 | Idle model-admission boundary |
| S5I5 | 609.53 | 63 | 47 | 0 | Host wall budget after action reconciliation |
| CD82 | 619.11 | 39 | 15 | 0 | Host wall budget after action reconciliation |

All three received exactly 600 seconds of model-call admission. Final admitted
requests completed 17.81 / 9.00 / 18.58 seconds after that cutoff respectively,
inside the reserved completion allowance; no later call was admitted. The
200-action caps were not reached. All final game states were NOT_FINISHED.
Executed actions come from game-progress and host-received action records, not
submitted action lists: the latter include actions later skipped by batch stops.

## Time and continued work

- Notebook last log: **2287.842 seconds (38m 07.84s)**; this is not an exact GPU
  billing measurement or queue-inclusive duration.
- Outer runtime: 2242.663 seconds (37m 22.66s).
- Sum of game sessions: 1846.880 seconds (30m 46.88s).
- Difference between notebook last log and game sessions: 440.962 seconds
  (7m 20.96s), covering bootstrap, verification, serving setup, teardown and
  notebook finalization, not merely model loading.
- 153 Actor calls: mean 11.80s, median 10.13s, linear p90 21.81s, maximum 35.98s.
  Their recorded wall durations total 1805.377 seconds, 97.75% of game-session
  time. This is end-to-end client-call time, not isolated GPU decoding time.

| Game | Calls admitted, first / second 5 min | Actions observed, first / second 5 min | Actions observed after 600s |
| --- | ---: | ---: | ---: |
| LS20 | 33 / 18 | 3 / 3 | 0 |
| S5I5 | 35 / 28 | 6 / 40 | 1 |
| CD82 | 24 / 15 | 3 / 11 | 1 |

Windows begin at admission_deadline minus 600. Calls are grouped by admission,
so their whole durations can extend beyond the window; actions are grouped by
when the host received their successor. This proves continued execution later
in each session, not continued progress toward winning.

Maximum prompt tokens were 68,381 / 131,890 / 84,622. Mean call duration grew
from first to second window: 8.96 -> 17.48s, 8.41 -> 10.46s, 12.73 -> 20.16s.
Prompt growth and longer calls co-occur here; this run does not isolate causality.
Worker peak RSS sampled at model boundaries was 238.98 / 241.96 / 235.18 MiB.
Those are process peak measurements, not exact final peaks or GPU memory.

## Verification and evidence

[audit.py](audit.py) completed successfully; [AUDIT.json](AUDIT.json) preserves
its checks and descriptive statistics. It verifies:

- Frozen build and downloaded runtime file identities, adapter/shared-solver
  identities, and remote notebook source equality.
- Three exact public game allocations, action counts, request IDs and deadline
  ordering, no failed/unfinished request lifecycle, and reasoning/summarizer off.
- 153 completed solver requests plus one runtime warmup: 154 successful chat
  accesses in the whole server log, no unmatched accesses.
- Clean TERM-only server exit, return code zero, lifecycle containment passed,
  no remaining requests; no OOM, deleted-request state, server traceback or HTTP
  4xx/5xx markers. Notebook logs were also checked for these failure categories.

All 178 output files are retained in raw/. Primary evidence:
[game progress](raw/qualification/game-progress.jsonl),
[allocations](raw/qualification/budget-allocations.jsonl),
[outer result](raw/qualification/result.json),
[server log](raw/qualification/server.log),
[terminal notebook log](raw/arc3-first-submission-20260910.log), and each session's
ipc/request_lifecycle.jsonl, model_calls.jsonl and execution_timing.jsonl under
raw/qualification/solver-workspaces/.

## Decision

Retain the candidate's public operational qualification. The longer run answered
whether it continues making calls and executing actions through ten minutes per
game: yes. It did not demonstrate a level win in any tested game.

Do not extrapolate a fixed actions/hour rate to nine hours: the game differences,
prompt growth and changing call durations are already visible. The next useful
investigation is the preserved decision/tool traces behind zero completed levels,
not another endurance run by default. No new experiment or code change is made
by this report. Current quota display: 8.06 GPU hours remaining; the before/after
rounded difference is 0.64 hours, not precise accounting.
