# Real-model throughput on the shared route — bears on the 200-decision cap

Source: `pranavagrawai/prototype-nooa-upstream-public3`, allocated 2026-09-05T12:17:56+05:30 after
8h10m queued, terminal ERROR at 15:08:50. Output downloaded before interpretation. This is the
reference arm, not the q6 candidate, but it shares the model, the SGLang server, the FP8 route and
`runtime/local_solver.py`.

## Hardware and model were never the problem

`NvidiaRtxPro6000` was genuinely allocated — the 2026-09-03 P100 mis-allocation did not recur.
SGLang launched at t=279 s and warmup completed at t=426 s (147 s). `shutdown.json` reports
`status: passed`, `containment_passed: true`.

## Observed decision rate

Three games ran concurrently against one server, `public_max_actions: 40` each.

| Wall clock | Elapsed | Event |
| --- | --- | --- |
| 06:50:15 | 0 | three games created, agents start |
| 06:57:02 | 7 m | s5i5 action 1 |
| 07:10:23 | 20 m | cd82 action 1 |
| 07:17:32 | 27 m | ls20 action 1 |
| 07:42:02 | 52 m | all three action 2 |
| 08:05:02 | 75 m | s5i5, cd82 action 3 |
| 08:05 -> 09:35 | 90 m | silence, then decision timeout |

Solver decisions actually submitted, from `official-launcher-result.json` `solver_sessions`:

| Game | submitted_batches | env actions | levels |
| --- | --- | --- | --- |
| cd82 | 1 | 4 | 0 |
| ls20 | **1** | 2 | 0 |
| s5i5 | 3 | 3 | 0 |

**Five solver decisions in 2 h 45 m of gameplay. On ls20 specifically: one.** The run then died at
`local_solver.py:392` with `TimeoutError: worker timed out waiting for decision`
(`launcher_returncode: 70`, `returncode: 71`, `timed_out: false` — the launcher's own 25200 s limit
was never reached; the worker stopped answering).

## Why this bears on the preflight

The preflight caps the run at **200 Solver decisions on ls20** inside a 7200 s gameplay launcher
timeout and a 14400 s session. Against the only real-model measurement available on this route,
that target is off by roughly an order of magnitude, even granting a threefold speedup from running
ls20 alone without contention.

The card's own invalidating condition — *fewer than 15 agent attempts* — is the binding one. At the
observed rate the run would not reach 15, and the likely terminal state is the same worker decision
timeout, which crashes rather than writing complete session artifacts. The card treats that as
**invalid, not partial success**.

## The assumption underneath the cap

q6 qualification (`analysis/local-probes/20260905T-stall-authority-q6/qualification/20260905-final-contract/SCORE.json`)
records `"platform": "scripted"`, `"model_calls": 0`, `"model_wall_seconds": 0`, `"wall_seconds": null`,
with `agent_turns: 11` and `steps: 51`. The candidate was qualified against a stub. **No real-model
throughput measurement for the q6 candidate exists.** The 200-decision cap was set without one.

This does not prove the q6 candidate is as slow as the upstream chassis — it is a different actor
and the number remains unmeasured. It does mean the cap rests on an assumption that the first real
measurement on the shared route contradicts.

## Bearing on the q6 question itself

The 90-minute terminal silence is a stall of exactly the kind q6 exists to study. Read one way that
is encouraging for stall incidence; read another it is the failure mode that prevents a run from
producing evaluable follow-up horizons at all. The preflight as configured cannot distinguish these,
because the stall that supplies the evidence is also the stall that invalidates the run.

## Status

One batch GPU slot is free. The push was not made. `maximum_pushes: 1` remains unspent and the
frozen bytes are unchanged. `prototype-tycho-upstream-public3` is still RUNNING and will supply a
second real-model data point on the other chassis at no additional cost; watch task `bpt2jxgf3`.
