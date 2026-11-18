# q6 quick host probe — the repaired host qualifies on Kaggle FP8

`pranavagrawai/q6-quick-host-20260905`, pushed 19:27:46 IST, queued 2 h 39 m, terminal **COMPLETE**
22:11 IST. Run 16:22:31–16:38:58 UTC, `elapsed_s` 986.0. Genuine `NvidiaRtxPro6000`.

`assess.py` returns **PASS with zero errors**. Every failure it reported against the speed
artifact is gone:

| assessor error on the speed probe | now |
| --- | --- |
| runtime/launcher did not pass | `status: passed`, `rc 0`, `timed_out: false` |
| wrong contract or arm | correct contract, control arm |
| changed grid falsely labelled no_impact | absent |
| missing `ipc/state_delivery.jsonl` | present and populated |
| official launcher Solver session schema differs | `nooa-join-sessions-v4` accepted |

## The reconciliation that killed the speed probe now holds exactly

```
expected_launcher_solver_requests  53
observed_launcher_solver_requests  53
whole_model_request_count_verified true
```

54 total chat completions, 1 warmup, 53 solver requests, 53 recorded actor calls. No unaccounted
request. This is the defect that ended the 2026-09-05 speed probe, fixed and confirmed on hardware.

## The repaired host is also markedly faster

Same game, same cap, same 10 executed actions, same 9 grid changes.

| | speed probe | quick probe |
| --- | --- | --- |
| status | invalid | **passed** |
| gameplay wall | 1332 s | 986 s |
| model calls | 106 | **53** |
| model wall | 1325 s | 669 s |
| decisions (turns) | 6 | 8 |
| per-turn wall, median | 140.6 s | **44.3 s** |
| per-turn wall, max | 720.2 s | 251.4 s |
| calls per turn | 11,19,11,4,**56**,5 | 6,16,12,3,3,3,7,3 |

Half the provider requests for the same work, and no 56-call excursion. Consistent with the HUD
repair no longer declaring a moving action inert and the dispatcher no longer re-delivering stale
state — both would have forced re-deliberation.

Integrity clean: 0 reasoning tokens across all 53 calls, `stop.json` reason
`competition_agent_cleanup` at 10 steps, so the run ended on its action cap rather than any timeout.

## Revised arm sizing

At the observed rates, inside the 7200 s gameplay window the earlier preflight assumed:

| basis | decisions in 7200 s |
| --- | --- |
| median 44.3 s | ~162 |
| mean 123 s (wall / 8 turns) | ~58 |

The 200-decision cap is no longer off by sixfold as the speed probe implied; at the median it is
within about 1.2x. It is still not established as reachable, and the mean and median differ by
nearly threefold, so the estimate is fragile.

## What this run does not establish

**The summarizer fix is unexercised.** `summarizer_calls: 0` — context never grew to the ~102k
compression threshold in 8 turns, so no compression call occurred. Reconciliation balanced at 53
precisely *because* nothing bypassed the recorder. The recording repair remains proven only by local
tests against a fake model. A longer run is the only thing that will exercise it, and a longer run is
exactly where the speed probe's 102k context and turn-4 excursion appeared.

**Throughput may not extrapolate.** Eight turns, capped at 10 actions, is too short to show whether
late-game context growth still degrades per-decision cost. The speed probe's mean was inflated by a
single pathological turn; this run simply never got long enough to have one.

**No stall claim.** Zero stall eligibility at ten actions under the unchanged q6 threshold, as the
card states. Levels completed 0, `NOT_FINISHED` — consistent with every other chassis on this route.

## Cost

Under one GPU-hour. Note the kernel writes `submission.parquet` into its output; no competition
submission was made.
