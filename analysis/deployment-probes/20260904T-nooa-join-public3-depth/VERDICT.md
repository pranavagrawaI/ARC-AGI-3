# Verdict — 20260904T-nooa-join-public3-depth

Kernel `pranavagrawai/prototype-nooa-join-public3-depth` version 1. Status ERROR. Elapsed 5350 s (~1.49 h). No retry of this card.

## Probe

**Repaired.** `official-launcher-result.json` has `public_progress_error` null and `solver_sessions_error` null. Schema `nooa-join-public-progress-v2`. Recording `action_input.id` remaining 0 did not mark the run as no play. Kernel ERROR is `subprocess.run(..., check=True)` on probe rc=71 (`residual_group_term_required` after the actor thread died).

## Play (truncated by crash, not by the 80-action cap)

| game | recording frames | grid_hash_changes | IPC env actions | max_levels | last IPC |
|---|---:|---:|---:|---:|---|
| ls20 | 45 | 44 | 45 | 0 | LEFT/RIGHT |
| s5i5 | 51 | 50 | 55 | 0 | CLICK 32 32 × many |
| cd82 | 35 | 31 | 35 | 0 | SPACEBAR/UP |
| **set** | **131** | **125** | **135** | **0** | |

None of the three reached 80. Chassis pin `q3c-mdfiles-after` on all three sessions.

s5i5 frame 49 is `GAME_OVER`; frame 50 is `NOT_FINISHED`. `available_actions=[6]` for the recorded tail.

## Crash

```
LocalSolverError: turn does not match the last public observation
runtime/local_solver.py:351  _validate_turn_continuity
```

`MyAgent.choose_action` returns `GameAction.RESET` on `GAME_OVER` without a Solver decision. The next `NOT_FINISHED` frame is a new grid. `LocalSolver.decide` still holds the GAME_OVER observation and rejects the turn.

v2 (16 actions) never hit GAME_OVER, so it never hit this. The existing adapter test `test_game_over_reset_does_not_let_the_official_backstop_truncate_solver_cap` uses `FakeSolver`, which does not check continuity, so it stays green.

## Card

Stop: success if any game `max_levels_completed >= 1`. Futility if 80 actions complete and all three stay at 0. Timeout/truncation: score the completed games; do not retry 80.

- Level ≥ 1: no.
- 80 actions completed: no.
- Truncated play: yes. 0 levels on 45/51/35 frames.

Result map: **truncated, 0 levels.** Do not retry this card. Do not spend 300–400 actions on these three until GAME_OVER resyncs the Solver. ls20 had 45 frames and still 0 levels; local q3c finished L0 in 32, so this is not only action-starved.

## Claim ceiling

The q3c join on R8-L3B can play past 16 actions and move grids at 80-cap. It does not establish that 80 actions complete a level. It establishes that GAME_OVER followed by the adapter RESET desynchronizes `LocalSolver` and kills the official loop.

Outputs: `analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/`.
