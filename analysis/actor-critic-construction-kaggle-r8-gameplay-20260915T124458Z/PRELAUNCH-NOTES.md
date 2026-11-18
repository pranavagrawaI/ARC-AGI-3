# Pre-freeze notes for actor-critic-construction-kaggle-r8-gameplay-20260915T124458Z

Written 2026-09-15 by the worker after the independent package review
(`analysis/actor-critic-construction-kaggle-verification-r8/repaired-review/DECISION.json`,
APPROVED_LOCAL_PACKAGE, seven findings). Not a card edit. Nothing frozen or launched.

## Findings acted on

- Finding 1 (bytecode pollution): both `__pycache__` directories removed; root cause closed in the
  suite itself — every test subprocess that imports from the deployed package or the tests directory
  now sets `sys.dont_write_bytecode = True` before importing (`test_teardown.py` scripts ×3, the
  e2e wrapper in `test_runtime.py`, the wrapper in `test_gameplay_arms.py`). Proven by running the
  full suite with `PYTHONDONTWRITEBYTECODE` deliberately unset: 102 passed, zero `__pycache__` after.
- Finding 3 (asymmetric call cap), by PM ruling: `ARM_CALL_CAP` 129/129, `TOTAL_CALL_CAP` 258,
  `local_accounting.MAX_ADMITTED` 258, `MANIFEST.caps` (`arm_call_cap` 129/129,
  `calls_total_including_warmup` 259), `launch_control.LIMITS` (`server_chat_calls` 259,
  `application_calls` 258, `arm_call_cap_a/b` 129), `test_frozen_parameters_match_card` asserts
  symmetry and that the total equals the sum of the arm purses; fixture caps in the e2e test 13/13.
  Card text: the Cap bullet "≤ 260 including 1 warmup (two arms at ≤ 130 each)" remains a true
  upper bound and was not edited; the one number in the worker-written Build section ("≤ 259
  admitted + 1 warmup") was corrected to "≤ 258 … + 1 warmup = 259" so the card does not state a
  purse the package no longer has. Contrast and Invalid-if untouched (finding 2 awaits the ruling).

## Finding 6 — arm B funding slack: ACCEPTED, envelope unchanged

Arithmetic (driver `--timeout-s` = 1800 − 60 = 1740; `admission_budget(remaining, arms_left, 90)`
= (remaining − 180)/arms_left − 30 − 90 − 15, must be ≥ 600 or the driver raises `cannot fund`):

- arm A at t≈0: (1740 − 180)/2 − 135 = 645 ≥ 600 → funded.
- arm B at t = t_A: (1740 − t_A − 180) − 135 ≥ 600 ⇔ **t_A ≤ 825 s**.
- t_A = setup + admission + retirement: setup (framework copy of ~80 files, preflight, seal, GPU
  query, ARM-DIFF) took ≈ 2 s in the local fixture and the R7 remote preflight files were written
  within seconds of launcher start; admission is exactly 600 s; the post-admission tail is bounded
  by the last in-flight request (`ARC_LLM_TIMEOUT_SECONDS` 90) + 15 s tool margin + worker
  retirement/cleanup (seconds). Expected t_A ≈ 600 + ≤105 + ~20 ≤ ~725 s → **≥ 100 s slack**;
  worst plausible case still leaves ~80 s.
- Why accept rather than tighten: the 180 s final-output reserve and the 30 s per-arm cleanup
  reserve are v10's tested policy; shaving them buys slack at the price of a driver-policy change
  the review did not see. If the slack is ever exceeded the failure is closed and legible:
  `RuntimeError: remaining launcher budget cannot fund the precommitted admission window` before
  arm B admits any call, arm A's row in `game-progress.jsonl` and receipts survive, and the run is
  reported as a funding failure, not a gameplay result. Result map: that outcome is "invalid —
  preserve, diagnose, new identity" with the reserve as the named fix.

## Finding 7 — post-arm invariants abort before arm B: NAMED, by design

After each arm the driver raises (and the supervisor reports `official launcher command failed`)
if any of these hold: `reconcile_official(...)['reconciled']` is false (recording file count ≠ 1,
recorded frames ≠ executed actions, final state/levels disagree with the last frame, or the
scorecard run state disagrees); the stop reason is not in {win, step_budget, wall_budget,
model_admission_budget, call_cap}; or arm A shows a freeze or a construction request. On such an
abort arm B does not play. Reading rule for the run: an `official-launcher-result.json` with
`status: failed`, `error` naming one of those conditions, and `games` containing only arm A is the
guard firing on arm-A bookkeeping, not a crash; the raw receipts, `game-progress.jsonl`,
`ARM-DIFF.json`, arm-A workspace and recordings are all retained for diagnosis.

## Frozen hashes at the time of these notes (post-ruling rebuild)

- package `construction.ipynb` sha256 d97079e385a90e2f23aee54081d838663d22c5dc86bd20d8cbdddf8bd07e153b
- candidate/package `MANIFEST.json` sha256 b721f5cb913499f63cea73d124b30949373a8706c433ad48cc8d2814ffb5a970
- card sha256 d38f90dca27bd840df2ecf0a5e2211af099bf9f7297ac6c35487c806128db881 (after the finding-2 amendment)
- gate `launch_control.py` NOTEBOOK_HASH / LOCAL_MANIFEST_HASH retargeted to the two above.
- The prior DECISION.json binds the pre-ruling hashes (33f0bfbd… / 8ff1505f…); a re-bind of the
  review is required before `make_manifest.py` will accept the identity.
