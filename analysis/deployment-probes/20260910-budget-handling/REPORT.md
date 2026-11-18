# Budget handling — 10 September 2026

Implemented and locally qualified in `deployment/kaggle/submission-20260910-v5`,
using runtime `submission-runtime-20260910-v3`. No Kaggle upload, GPU run or
competition submission was made. Prior candidates and raw runs remain unchanged.

Runtime contract: `8cb520c9b5070a9e50ed898d57461fbb6530d1a863d57b9a67528ea32374d6c1`.
Notebook SHA256: `b4ff2fc1e8ba3b7c916992252c1e942dff72db24a1f1bbd75014dd09051e1661`.

## What changed

Before every game, admission time is recomputed as
`(remaining gameplay seconds - 180) / remaining games - 30 - request timeout - 15`.
Unused time carries forward. The 180/30/15-second reserves cover final accounting,
per-game cleanup, and post-request tools/protocol work respectively. They are
policy choices, not empirically established worst-case bounds. An insufficient
allocation is rejected rather than silently oversubscribing the run.

At 25140 seconds, 110 games, and 90-second client timeout: about 91.9 seconds of
model admission, 105 seconds of request/tool reserve, and 30 seconds of cleanup
per game initially. Public notebook mode retains a 75-second admission cap;
competition mode uses the rolling allocation without that cap.

The absolute deadline includes worker startup and analysis. It is checked before
every Actor model call, not merely after a game action. The Solver sees remaining
admission time in its game context. On exhaustion it writes a marker and parks
without a new HTTP request. The worker acknowledges that idle boundary, the host
finishes normally, and the game loop continues without inventing an action.
The worker watchdog shrinks to remaining game time on later decisions.

Allocations, game outcomes, and request admission/completion/failure events are
preserved separately. Accounting rejects failed, unfinished, late-admitted or
unrecorded calls, including in budget-stopped games. Zero calls/actions are
allowed only with an explicitly acknowledged budget stop. The outer runtime's
request reconciliation was updated as well, not just the inner launcher.

## Verification

Final suite: **23 passed in 94.68 seconds**, using the exact packaged runtime.

- Budget arithmetic for 1, 3, 25, 55, 110 and 135 games; invalid-input rejection.
- Notebook payload and frozen-runtime hash checks.
- Actual official Agent loop, local offline LS20: 10 actions / 10 model requests.
- Same through a local ARC HTTP gateway: 10 actions / 10 model requests.
- Zero-admission stop: no model request and no invented action, in both modes.
- Two successive offline LS20 invocations both exhausted cleanly, proving the
  first budget stop did not abort the remaining loop. This is a repeat-game
  lifecycle fixture, not evidence from two different games.
- Multi-call analysis with no action: 8 completed requests, then clean exhaustion,
  in each mode. No ABORT markers; request accounting reconciled.
- Negative accounting fixtures reject unfinished, failed, late-admitted and
  missing-call records; outer zero-call accounting requires an explicit stop.

See the physical fixture directories in [local-v4](local-v4/), especially
[offline thinking result](local-v4/test_real_entrypoint_thinking_0/RESULT.json),
[gateway thinking result](local-v4/test_real_entrypoint_thinking_1/RESULT.json), and
[continue-after-exhaustion result](local-v4/test_real_entrypoint_True_Fals0/RESULT.json).
The outer reconciler was additionally applied to all six physical fixture
results: counts 10, 10, 0, 0, 8, 8. Runtime freeze verification passed.

Existing real contingent-worker regression: **4 passed in 21.70 seconds**, with
two pre-existing unknown `asyncio` mark warnings in an imported test module.
No matching worker/headless process remained in the post-test process scan.

Earlier local stages are retained: [local-v2](local-v2/) passed 15 checks and
[local-v3](local-v3/) passed 17. Initial collection failed because a test parameter
used pytest's reserved name `request`; it was renamed. A separate readback count
assertion initially counted pytest `current` symlink aliases as extra fixtures;
the corrected readback excludes aliases. Neither was a solver failure.

## What is not established

This is an operational budget policy, not a learned estimate of solving difficulty
or an optimal allocation. It does not establish Qwen quality, improved scores,
parallel throughput, or live SGLang shutdown behavior. HTTP timeouts, unbounded
tool execution, startup and cleanup failures remain failures, not budget stops.
In particular, the 15-second tool margin does not impose a hard sandbox bound.

The next warranted external step is a separately bounded public Kaggle
qualification of this exact candidate. It has not been launched or submitted.
