# Objection closure public run v1 result

## Verdict

**Invalid** under the card: the kernel ended in ERROR after LS20; S5I5 and CD82
received no allocation ("shortened game allocation"), and the launcher reported
a cleanup failure. No solver performance interpretation is permitted.

The objection contract itself behaved as specified on the one game that ran,
and the critic cost fell by an order of magnitude. Those are descriptive
observations on an invalid run, reported because they are the mechanism the
card set out to observe.

## Operational timeline

- Pushed 10:19 UTC; QUEUED for 100 minutes (v8 was RUNNING on the first
  monitor tick); RUNNING 12:00; ERROR at ~1,089s of kernel time.
- Model server ready; LS20 admitted at kernel ~555s with the frozen 600-second
  window. 42 model requests admitted, 42 completed, 0 failed.
- LS20: 7 actions executed, 38 Actor calls, 4 critic calls, 0 levels.
- At session ~466s (152s of admission remaining, 193 actions unspent) the
  Actor's final cell called `return_result(RespondReason.DONE, ...)`. The
  worker stopped answering; the host's `decide()` waited 120s and raised
  `TimeoutError: worker timed out waiting for decision`; the launcher exited
  rc=1; the notebook cell raised; Kaggle marked the kernel ERROR.

## First failing boundary

The Actor declared the game over. The harness instruction it was given reads:
"If the game is stuck/over (harness note says so), summarize what you learned
and return_result(RespondReason.DONE, ...)". No harness note said so. The
state was NOT_FINISHED with 152s remaining.

Three layers, in order of evidence:

1. **Inherited fragility, v8 chassis (established).** `headless.py` accepts a
   DONE on a NOT_FINISHED game and only prints it to stderr. The worker then
   waits for a notification the host will never send, because the host is
   waiting for a decision. One Actor DONE therefore ends every remaining game.
   v8 never exercised this path; v9 did not change it.

2. **New in v9 (established).** critic:4 read the packet's
   `affordable_actions_at_this_rate: 2` (215s remaining ÷ ~93s per action) and
   asked: *"With only two wall-clock actions remaining, is the current LEFT
   test cheaper than directly applying the remaining moves to clear the
   level?"* — a projection stated as a fact, plus a prescribed plan. The
   Actor's next call: *"2 wall-clock actions left. Can't solve in 2 moves.
   Close objection: pivot to execution."* It withdrew (retiring LEFT),
   submitted two more "final" actions, and then DONE. The arithmetic was
   correct; its effect was to make the Actor stop rather than act faster.

3. **Unexplained (not established).** From turn 4, before critic:4, the Actor
   wrote "Budget exhausted (100/100)". No host surface produces that string:
   not the state note, observations, the chassis, or the NOOA package; v8's
   Actor, which saw the same "Budget stops further model calls … prioritize
   executable progress" status line, never wrote it. Tool results were not
   preserved, so a confabulation cannot be distinguished from an unpreserved
   surface. It is recorded as unexplained.

## Contract events on LS20 (descriptive)

| event | count | detail |
|---|---:|---|
| critic reviews | 4 | all `completed`; 0 invalid, 0 `restated` |
| objections opened | 3 | critic:2, :3, :4 (critic:1 asked nothing) |
| dispositions | 3 | probe (1 action), narrow, withdraw (retired LEFT) |
| reminders before disposition | 0 | every objection closed on the next commitment |
| probe accounting | 1 | `probe_exhausted` fired at the deadline step |
| recurrence rejections | 0 | |
| expiries | 0 | |
| `closes` payloads rejected | 0 | |

Every predicted contract behaviour that could be observed on one game was
observed. The "no stall" prediction failed in the worst way available: not a
stall but a self-declared end.

## Critic cost on LS20 (descriptive, n=1, against v8's LS20)

| metric | v8 | v9 | card prediction |
|---|---:|---:|---|
| critic wall share of session | 26.0% | **1.3%** | < 8% |
| mean critic call | 18.2s | **1.5s** | < 8s |
| critic prompt tokens | 32k → 117k | **1.7k–2.2k** | < 8k |
| critic ÷ concurrent Actor prompt | 1.13–1.32× | **3.6–8.1%** | < 15% |

## What the returned wall bought

Nothing, on this run. LS20 executed 7 actions in 466s (66s per action) against
v8's 17 in 625s (37s per action). The Actor made 5.4 calls per action against
v8's 3.7. The critic's cost was gone and the Actor spent the time inspecting,
then quit. The "executed actions exceed v8" prediction is refuted on the one
game that ran, for a reason the card did not anticipate.

## Result map

Invalid evidence → no solver performance interpretation. Diagnosis of the first
failing boundary is above. The permitted next action is a runtime that (a)
refuses a DONE on a NOT_FINISHED game with admission remaining and (b) stops
handing the critic a projected action count to state as fact — under a new
card. No relaunch under this card.

## Evidence

- `raw/qualification/solver-workspaces/ls20-9607627b-bsa1cz1e/ipc/`:
  `model_calls.jsonl` (call 41 is the DONE cell), `objections.jsonl`,
  `critic_reviews.jsonl`, `request_lifecycle.jsonl`, `actions.jsonl`.
- `raw/arc3-objection-closure-20260912.log`, `live-log.txt`: the launcher
  traceback at kernel time 1075s.
- `monitor.jsonl`: 154 ticks; QUEUED 10:20–12:00, RUNNING 12:00, ERROR by 12:19.
- `PRELAUNCH.json`, `push.log`, `postrun-status.log`, `postrun-quota.log`.
