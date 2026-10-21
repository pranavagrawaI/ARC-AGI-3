# Guard review fixes, 2026-10-21

Local corrections to P1/P2 in `analysis/guard-review-20261021/review.md`. No GPU, network mutation, launch, submission,
queue/timer edit, staging or commit. Production-timing CPU jobs owned elsewhere were not changed or terminated.

## Token accounting

The old end guard counted only `GameRun.final_generated_tokens`, which covers the final turn without a move. The actual
solver generally leaves that field zero; previous action tokens are in `history[].generated_tokens`. It therefore rejected
three genuinely played F1x screen-2 zero-level games as an unserved zero. The new expression sums action-history tokens
and final-turn tokens once per game. Zero-token RESET records do not establish model activity.

The regression initially reproduced `RuntimeError: ... 0/0 served games ... 0 tokens, no level`; a second regression
expected 10 tokens from a 7-token action plus a 3-token final turn and observed only 3. Both now pass. A separate explicitly
synthetic dead-endpoint fixture with a zero-token RESET and analyzer-failure termination still raises.

`tests/fixtures/f1x-played-zero.json` preserves only the real guard inputs of the three recorded games, plus source path,
source SHA256 and game IDs. The full source benchmark is untracked and is not needed to run the tests. Its SHA256 is
`59a1285e7a08385c9bece108f4ca5544ad2f2f972ac54d8446fe0e5c9cdde63a`; selected IDs are `sk48-d8078629`, `dc22-fdcac232`,
`cd82-fb555c5d`. The fixture retains all 140 action token records, totaling exactly 233,500 tokens, and zero final-turn tokens.

## Launch freeze

`build.main` now refuses at entry whenever `experiment/fault-screen-1/launch-attempt.json` exists, before any output mutation,
including control-only and empty requests. This replaces the old branch that verified one notebook but then removed the
fault-screen manifest entry, causing the next build to fail with KeyError.

The regression runs the real builder in a temporary copy, writes a launch marker, then requests F1x twice, control and an
empty build. Every request now raises the explicit freeze error and leaves the entire byte inventory unchanged. A separate
test verifies actual repeated prelaunch builds produce identical files and retain `fault_screen_1`. The old code reproduced
both the non-refusal and the second-build KeyError; logs are retained.

## Verification and current freeze

`python3 -B -m unittest solvers/m2_guard/tests/test_guard.py`: 18 PASS, comprising original 13 and five added regressions.
`git diff --check`: PASS. Red/green logs, pre-edit build/card snapshots, final builder output and hashes are in this directory.
Parent/root and the independent guard reviewer also ran the 18 tests; the reviewer independently regenerated the notebooks.
The compact fixture portability correction was then supplied for a final provenance recheck.

All three generated notebooks were rebuilt locally before any launch attempt. `manifest.json`, the fault-screen card and
`freeze.json` match the generated notebook hashes. Fault-screen notebook SHA256 is
`0de155058ecab4e0264aa19a1b993ebb9bff3382abe70040905e97403ee45480`; metadata remains
`71eef644b5fa085b9fbea0ed4ef27c1a45498d28728c2cb673bc821c443b8aec`. Status is PREPARED_NOT_LAUNCHED with no launch marker.

The card no longer treats one CPU timing/configuration as proof against every hidden startup-failure cause or pending
production-timing jobs as completed evidence. Recovery from a live but unusable server remains unsupported: after first
READY, the supervisor watches process exit only. The process-kill screen cannot qualify live-health/request-failure recovery.

## Bounded risk comparison

`python3 -B solvers/m2_guard/evidence/review-fixes-20261021/measure_risk.py after` runs the frozen original 13 test IDs;
`... measure_risk.py expanded` runs all 18. `risk-before.json` was captured before the production fixes. The estimator
reuses existing repository AST/stdlib-trace rules for changed `build.main` and the embedded end guard only. It introduces
no dependency, exclusion, threshold or gate. Embedded lines use an explicit filename and are offset by one for the estimator's
function wrapper. The first invalid untraced estimate is preserved as `risk-before-untraced.*`; it is not the baseline.

Under the identical original 13-test universe, main CC/coverage/CRAP remain 5/0/30. Embedded end guard CC/CRAP rise from
14 to 16 at full statement coverage in both revisions; the added branches account for required history-token summation.
This is a correctness fix, not a claimed risk-score improvement. The expanded 18-test measurement covers both changed
units completely: main CRAP 5, end guard CRAP 16, maximum 16. It is separately labelled and is not compared as an improvement
against the old-suite baseline. No threshold or violation count applies; no GPU/supervisor-runtime coverage is claimed.

Pre-mortem cases addressed: played zero mistaken for unserved zero; final-turn tokens lost or counted twice; RESET history
mistaken for model activity; first postlaunch build mutating the freeze and second build failing; control-only requests
bypassing the freeze. All have regressions or direct byte-inventory verification. Live-server recovery and hidden failure
attribution remain explicit scope limits.
