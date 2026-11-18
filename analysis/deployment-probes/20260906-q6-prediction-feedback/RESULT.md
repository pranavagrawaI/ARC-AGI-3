# q6 prediction feedback: local result and longer-run handoff

The final local repair replay passed in two fresh starts. Each used17 model calls, saved a repaired executable model, and submitted one legal next action. The exact helper source frozen for that accepted action predicted all independently evaluated fields correctly on all four held-out transitions; the encoder also matched the independently measured grids. The final contract is e9b0cae6d1b4e0a1c7224d9e34b3d57a0d0bbe8985764e7bfa9bd443b80bddbf.

This establishes that the repair can carry through to an action on this regression case. It does not establish a reliable three-call workflow, lower latency, improved level solving, or generalization to unseen games. The held-out frames were withheld from these Actor starts, not from the test designer. The replay stops at submission: it does not pretend to execute an unsupported successor for the chosen action. Actual sustained gameplay is the separate Kaggle run.

## What changed

The isolated candidate freezes the Actor-authored encode/predict source before each admitted action and compares its predictions against actual successors under the same encoder. Field mismatches are visible in current-state feedback; a successful fresh check of the same helper version is needed before batching. Unresolved or absent models still allow single-action exploration. A short partial-repair/test workflow tells the Actor to correct independently disproven parts without first reconstructing the entire game.

We fixed an observed state-access trap: the original asynchronous queue getter returned coroutine objects when called without await. In the first live replay, q6 accumulated those objects in an unbounded loop and reached about21GB RSS before termination. The Actor now reads latest_state()/trajectory() synchronously; consuming the dispatcher's queue immediately raises an actionable error. Its CPU worker has a4GiB virtual-memory limit. Automatic pure-Python helper computation has a1s alarm budget, preserving existing signal handlers; native extensions may defer signals, so outer worker containment remains necessary. Incomplete final JSONL appends are tolerated, while complete malformed rows remain errors.

The longer-run adapter now stops admission cleanly at its action/time budget, reconciles the last actual successor, and preserves final accounting. A separate post-run replay evaluates the final pending prediction prefix without claiming the Actor saw that feedback.

## The unsuccessful tests matter

| Identity | Actual starts | Calls | Outcome |
| --- | ---: | ---: | --- |
| a |1 baseline |4| Coroutine accumulation; killed worker; no feedback comparison. |
| b |1 repaired baseline |12| No action; harness stopped whole comparison at one arm's cap. |
| c |2 baseline,2 feedback |48| No accepted actions; one feedback start saved a partial right-edge repair but introduced a wrong left-edge prediction. |
| d |2 workflow |24| No actions at12-call cutoff; one saved model passed4/4 measured held-outs. |
| e |2 workflow |30| Request-body cap censored starts at14/16calls; one saved model passed4/4, neither submitted. |
| f |2 workflow |34| Both submitted at17calls; both exact committed models and encoders passed4/4. |

Each revised candidate, cap, or harness policy had a fresh identity and precommitted card. Archives, raw requests/responses, local cap refusals, and source hashes remain under runs/. Earlier saved-helper results do not get promoted to action-committed success. Identity c's original scorer did not check a newly added band_left field; that defect is explicitly reported, and subsequent scoring checks it when present.

Total reported OpenRouter spend, including unsuccessful trials: $0.655591166 for152 responses. Conservative reservation accounting stayed below the unchanged cumulative$1 ceiling. No reasoning-enabled trial or inference-engine change was introduced. Increasing the local call/body envelope enabled completion; there is no matched baseline at that final envelope, so no causal speed/quality advantage is claimed.

## Verification and next run

Local regression/integration checks passed, including real CodeAct contradiction -> batch rejection -> helper repair -> correct single prediction -> batching reopened; captured raw-frame comparison; source freezing; resource/loop guards; malformed and partial-tail handling; real worker deadline cleanup; state/history delivery; Actor/summarizer accounting; and monitor terminal/download behavior. The notebook was rebuilt/extracted and its source hashes checked. The installed Kaggle client request verifier passed with zero transport calls.

The longer run uses the same candidate source, baseline FP8 inference, one ls20 game,100 individual actions maximum, and7200seconds of decision admission. An admitted decision may finish within the900second worker bound; launcher envelope8400seconds, Kaggle hard cap10800seconds/3GPU-hours. Stall-review treatment remains off: this run tests prediction feedback and sustained solver behavior, not the q6 stall intervention effect. No competition submission and no automatic push retries.

PUSH.json records the launch receipt. long-run/MONITOR.json records queue/run/terminal status. The detached monitor preserves a frozen candidate, downloads raw output once, runs host assessment, replays final pending feedback separately, and writes long-run/RESULT-LONG.md plus REPORT-LONG.json. A host PASS is separate from behavioral review; a failed or censored run remains visible.
