# Live OpenRouter capability probes

Executed on 2026-09-05 (the directory's 20260906 label is not the execution date). Thirty real requests to qwen/qwen3.8-27b, served by Reka, cost $0.085124674 according to response usage. Zero reasoning tokens were reported. The conservative cumulative reservation was $1.23452406, below the announced $1.50 cap. All runs stopped at their ten-call cap. No Kaggle package or source was changed.

| Run | Calls | Actions | Levels | Reported cost |
|---|---:|---:|---:|---:|
| Original control start | 10 | 0 | 0 | $0.025208 |
| Original treatment start | 10 | 0 | 0 | $0.028383 |
| Action-first kickoff variation | 10 | 1 | 0 | $0.031533462 |

The first two together cost $0.053591212. Control/treatment did not reach the 25-action detector floor, so these are not tests of q6's review effect. A ten-call start also cannot establish eventual game-solving ability. The model/provider and local environment differ from the exact Kaggle FP8 execution.

## Demonstrated

The model executed Python through the real CodeAct actor, read host observations, parsed a 64x64 grid, computed color counts and bounding boxes, rendered focused regions, and wrote hypotheses/step notes. The hypotheses were explicitly marked untested. It recovered from coroutine and missing-notification errors by using latest_state. With an explicit action-first kickoff, it submitted a legal UP action with both a verbal prediction and a typed predicted_path.

## What failed or remains unproved

1. **Slow transition from inspection to experiment.** Neither original ten-call start acted. The targeted kickoff requested an action within two calls, but the action arrived on call 8. Raw request 1 confirms that the modified instruction was present. This is observed weak prompt adherence in one directed start, not a general model ranking.
2. **Incorrect specific prediction concealed by a coarse success label.** The action-first run predicted changes only in rows 54–61, columns 2–9, identifying the bottom-left object as the player. Its UP action changed 52 cells, all outside that predicted region, with a bounding box of rows 40–62, columns 13–38. The host nevertheless scored `expect: move` as matched with accuracy 1.0. That score validates only the outcome class; it does not validate the claimed player identity, displacement or change region. No prediction revision was demonstrated before the cap.
3. **State-access friction partly belongs to the chassis.** The original control used the async queue as a synchronous value three times and later hit a code timeout waiting on it. In the directed run, call 9 processed a queued turn-0/step-0 notification after the UP action, while host artifacts had advanced to turn 1/step 1. Its output therefore showed empty action_results. Calling latest_state reads current state without consuming old queued notifications; the startup user message and state notification are separate dispatcher inputs. The trace supports stale notification exposure. It does not prove the model would fail to recover with more calls.
4. **q6 does not cover this early failure mode.** Its detector requires at least25 actions. Zero-action inspection cannot trigger it. Improving post-action authority review alone does not ensure the agent starts experimenting.

## Judgment and next move

The actor can inspect, code, record hypotheses and issue a legal predicted action. These probes do not demonstrate a reliable observe–predict–act–check loop. The immediate blockers are action initiation, unambiguous current-state access and checking the actual claim behind a prediction. A longer six-run arm would not answer those interface questions efficiently.

The next useful change would be a separately qualified state-delivery correction plus a behavioral test in which one action's specific predicted change is compared with the real next frame. Keep the queued Kaggle candidate unchanged unless deliberately replacing it with a new candidate/card. No fixes were made during these probes.

## Evidence and limits

- `SUMMARY.json`: per-run counts/costs, provider, reservation and exact prediction mismatch calculation.
- `requests.jsonl` / `responses.jsonl`: complete credential-free upstream requests and responses for the two original starts. Control calls2–7 show state-read errors/recovery; treatment call7 writes untested hypotheses.
- `../20260906-openrouter-action-probe/requests.jsonl` and responses: modified kickoff in request1; notes in response7; prediction/action in response8; old step0 feedback in request10.
- Worker session ipc files, notes, stderr and stop artifacts remain under each run's workspaces/.
- `PREFLIGHT.json`: live account/model check before running. No post-run account-balance delta was used; cost is the sum of each response's reported usage.
- Both cards were written before their corresponding first paid request. The local relay enforced call/input/output/aggregate-reservation caps and no retry after an upstream failure. Provider price ceilings use OpenRouter's documented max_price routing: https://openrouter.ai/docs/guides/routing/provider-selection .

The adapter's official base is represented by the previously tested local fixture, using the hash-pinned official frame conversion. These are real-model/local-game diagnostic probes, not a full official Kaggle launcher reproduction. End-of-probe ABORT is the deliberate cap, not an unexplained production crash. No paid retries or further probes remain running.
