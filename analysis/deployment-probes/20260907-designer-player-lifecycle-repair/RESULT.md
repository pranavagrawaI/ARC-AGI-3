# Lifecycle fixes: locally qualified, no paid recheck

Both requested failures are repaired in the isolated reasoning candidate.
No OpenRouter call, upload, Kaggle launch, or model/sampling/token-budget change
was made during this repair.

## Changes

1. `headless.py` propagates unhandled generation failures to worker supervision.
   It no longer logs a false retry and waits on a consumed notification. It also
   does not replay a notification that may already have submitted actions.
   This is fail-fast handling, not a new recovery/retry policy.
2. `shared_local_solver.py` aborts the protocol worker when a sent decision is
   interrupted, including by an outer signal. Termination happens while the
   model-admission slot is held. Subsequent adapter cleanup fails promptly on
   the closed solver instead of sending finish behind unfinished work. If
   retirement itself fails, the original error carries a diagnostic note and
   future model admission is blocked rather than assuming the worker stopped.
3. `openrouter_probe.py` now closes an independent, clock-checked admission window
   before timeout cleanup. The window does not acquire a lock held by upstream
   HTTP I/O. Existing transports remain attributable until completion or the
   bounded cleanup failure; failure stops the arm sequence. Relay shutdown is
   itself bounded and records unresolved retirement instead of hanging after
   the earlier cleanup bound. The old started run cannot be reused.

## Verification

- Full candidate suite: **53 passed**, one existing `set_dynamic` deprecation
  warning, 26.56 seconds.
- Old archived dispatcher against the new regression tests: **2 failed / 3
  passed**, as expected. The two unhandled-error cases time out in its false
  retry; the repaired dispatcher preserves the original exception immediately.
- Controlled real worker/adapter/HTTP probes: a length-limited reasoning response
  fails visibly; an outer SIGALRM is injected only after HTTP admission. Protocol
  worker and Actor child are non-executable before releasing a late response.
  No late action or replacement request is produced; actual adapter cleanup
  does not wait behind the unfinished turn.
- Those two probes also pass against notebook-extracted sources, not only loose
  Python files. Normal reasoning-to-action and state/successor controls remain
  green.
- Controlled admission tests reject expired/closed windows without relying on
  signal delivery, retain in-flight ownership, and bound blocked server shutdown.
  An injected retirement failure preserves the timeout and denies later admission.
- `prepare_candidate.py --check`: PASS. Mocked Kaggle CLI verification: PASS,
  zero network calls, competition `arc-prize-2026-arc-agi-3`, RTX PRO 6000.
- Independent scoped review found no remaining blocking regression after the
  shutdown-bound and cleanup-exception corrections.

Test command (with the documented Tycho/NOOA PYTHONPATH and reference Python):
`python -m pytest -q deployment/kaggle/designer-player-reasoning-20260907/test_*.py`

## Identity and evidence boundary

Current contract:
`6b8a5c1d6955c09d2f324f9ce806539fceec37254e9b7832de8f604dec89733e`.
The rebuilt notebook embeds the new dispatcher and transport. The source
manifest also binds the local probe's admission helper and runner.

The previous live contract `f26e4a957111...` and its failed result are unchanged.
Its complete pre-repair source archive is retained in the original OpenRouter
result directory and was verified against every entry in its original manifest.

Local failure handling is qualified. Live integrated qualification and remote
scheduler retirement are not inferred from these local tests. The next step is
a separately bounded live recheck; `launch_ready=false` and
`openrouter_qualified=false` remain honest pending that result.
