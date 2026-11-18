# q6 speed probe — decision latency measured on Kaggle FP8

`pranavagrawai/q6-speed-probe-20260905`, pushed 15:49:08, queued 1 h 34 m, RUNNING 17:25:57,
terminal ERROR 17:54:06 (28 m 09 s). Genuine `NvidiaRtxPro6000`. One push, no retry.

## Gameplay succeeded; the run died in post-run accounting

`stop.json`: `{"reason": "competition_agent_cleanup", "steps": 10}` — the probe **reached its
10-action cap cleanly**. The launcher agrees: `status: passed`, `returncode: 0`,
`launcher_returncode: 0`, `timed_out: false`.

The probe still reports `status: invalid`, from a step that runs *after* play:

```
File "/tmp/official_launcher_runtime.py", line 1376, in expected_launcher_solver_requests
    raise RuntimeError("official launcher Solver session schema differs")
```

| | reconciler expects | NOOA join emits |
| --- | --- | --- |
| `schema` | `tycho-nooa-public-solver-sessions-v1` | `nooa-join-sessions-v3` |
| totals | `total_actor_calls` | `total_submitted_batches`, `total_submitted_env_actions` |
| per game | `model_evidence.actor_call_count` | absent |

`exact_request_outer_runtime.py:1370` reconciles provider-request counts against the **Tycho**
session schema. The NOOA join emits its own. Any q6 NOOA-join run that finishes gameplay dies here,
**including the frozen q6-control-preflight**, which embeds this same v15 runtime. The preflight
would have played correctly for hours and then failed on this line.

Related in kind to the recorded finding that the Tycho scribe is not in the NOOA solver: Tycho-shaped
machinery pointed at the NOOA join.

## Measured latency

Gameplay wall 1332.1 s; model wall 1325.0 s. **99.5 % of gameplay is model time** — the route is
model-bound, not IPC- or environment-bound.

| Unit | n | min | median | max | mean |
| --- | --- | --- | --- | --- | --- |
| model call | 106 | 2.6 s | 8.3 s | 44.7 s | 12.5 s |
| decision (agent turn) | 6 | 48.5 s | 140.6 s | 720.2 s | 220.8 s |

6 decisions produced 16 submitted env actions, 10 executed before the cap. Per-turn model calls:
11, 19, 11, 4, **56**, 5 — turn 4 alone spent 720 s across 56 calls, and is what pulls the mean far
above the median.

Clean on the card's invalidating conditions: `reasoning_tokens` 0 on all 106 calls, every
`finish_reason` is `tool_calls`, 0 truncated batches, 0 force-advances, integrity coverage
137 COMPLETE / 62 UNOBSERVED.

## What this says about the 200-decision cap

| Decisions reachable in the preflight's 7200 s gameplay window | |
| --- | --- |
| at the median 140.6 s | ~51 |
| at the mean 220.8 s | ~32 |
| **200 decisions would need** | **~44,200 s ≈ 12.3 h** |

200 is unreachable by roughly sixfold against a 2 h gameplay window inside a 4 h session.

## Corrections to the pre-probe reading

Two inferences drawn from `prototype-nooa-upstream-public3` were wrong, and the probe is the
discriminating test that shows it.

1. **The preflight would clear the >=15-agent-turns bar, not fail it.** ~32-51 turns fit the
   window. The earlier claim that it would fall short was extrapolated from the upstream chassis and
   does not hold for this candidate.
2. **No stall occurred and no worker timeout fired.** The longest decision was 720 s, inside the
   probe's tightened 900 s worker timeout and far inside the preflight's 2400 s. The predicted
   upstream-style `TimeoutError` failure did not reproduce.

**Upstream slowness does not transfer.** The actor dominates the shared route: upstream managed 1
decision on ls20 in 2 h 45 m with 1 grid change across three games; this candidate made 6 decisions
in 22 minutes with **9 grid changes in 10 actions**. Reasoning from the shared model, server and IPC
was too pessimistic.

The conclusion that pushing the frozen preflight would waste the push survives — for a different
reason than predicted. The blocker is the schema reconciliation, not throughput or stalls.

## Stall incidence remains unmeasured

`stall_events: 0`, `evaluable_events: 0` at 6 turns. The scorer returns INVALID on
`fewer than 15 agent turns per game`, which is the probe's own 10-action cap doing its job, not a
defect. Six turns cannot estimate incidence, so the card's >=3-evaluable-stalls feasibility threshold
is untouched by this probe. Turn 4's 56-call, 720 s excursion is the shape of the behaviour q6 means
to catch, but the scorer did not classify it as a stall.

## Disposition

The q6-control-preflight identity and its single push remain unspent. Before any q6 run is worth
GPU time, `expected_launcher_solver_requests` must accept the `nooa-join-sessions-v3` schema, or the
join must emit what the reconciler expects. Fixing that changes frozen bytes, so it requires a new
preflight identity and card. The decision target should be set from the measured 140-220 s per
decision rather than from 200.

Cost: 0.47 GPU-hours. Quota after: recheck before the next push.

## Fixes applied 2026-09-05

Two defects, both required; either alone still fails the run.

**1. The reconciler did not know the join's schema.** `expected_launcher_solver_requests` accepted
only `tycho-nooa-public-solver-sessions-v1`. It now accepts `nooa-join-sessions-v4` as well, and the
join emits per-game `model_evidence` (`actor_call_count`, `actor_calls_by_sequence`) plus
`total_actor_calls`, derived from `ipc/model_calls.jsonl` and keyed by decision sequence. Counts for
a sequence repeated across episodes add, since a GAME_OVER reset restarts turn numbering.

**2. The recorder undercounted, which would have been the next failure.** `install_summarizer`
builds a separate Agent that inherits the solver's LLM but owns its own event manager, so context
compression reached SGLang without passing `_guarded_llm`. The probe proves it: prompt tokens ran
to 102,149 at call 85 and collapsed to 18,323 at call 86, and the server logged 108 chat completions
against 106 recorded plus 1 warmup. The interceptor is now installed on each summarizer too, and
records `summarizer: true`.

That second gap was also an integrity hole, not only an accounting one: `_guarded_llm` runs
`check_messages` and fails the run on reasoning tokens, and summarizer calls skipped both. Reasoning
tokens are an invalidating condition on the q6 card, so they could have appeared unseen.

**Replay against the real run.** Recorded as-is, 106 does not reconcile against the 107 solver
requests the server logged. With the summarizer call recorded, the join emits 107, the reconciler
accepts it and returns 107. The fixed path reconciles the run that failed.

**Ownership correction.** q6's builder embedded the *v15* candidate's `exact_request_outer_runtime.py`,
which is frozen against V14 bytes — patching it in place broke v15's freeze. q6 now carries its own
copy, as every other candidate does, hashed into the contract under `SOURCES`, which also closes a
gap where the contract did not cover the reconciler the run depends on.

Contract moved `1d65e366…` -> `62e4ad17228bfd9c230ece18e6cb96d6404470621068f9d33ed761a264952f4c`.
Arm notebooks rebuilt; they still differ only in `ARC_STALL_REVIEW`. New tests in
`deployment/kaggle/tests/test_q6_actor_call_evidence.py` (7, passing).

**The old preflight is now unlaunchable by design.** `q6-control-preflight` pins the retired contract
and refuses to build: *"preflight requires its qualified contract; use a new preflight for changes."*
Its two tests fail for that reason. A successor preflight needs a new identity and card, with a
decision target set from the measured 141-221 s rather than 200.

**Unresolved:** `ARC_LLM_CONTEXT_WINDOW=32768` did not bind — compression triggered near 102k, not at
80% of 32768 (26,214). Worth settling before sizing the arm, since it governs how often the
summarizer spends a request.
