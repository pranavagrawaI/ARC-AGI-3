# Actor beliefs — Tycho investigation

## Observed so far
- Level 0, turn 0, state NOT_FINISHED. Action limit 2/turn, actions: RESET, UP only.
- Frame level_0/turn_000.txt: 64x64 grid, ALL cells 0 (blank/white). No visible entities, no HUD pixels.
- No prior attempts, no transitions, no death/terminal events on record.

## Notes on tooling
- run_python is BROKEN in this environment (bwrap cannot exec the venv python). Cannot use
  wmlib/np analysis; must reason from inlined frames/diffs and on-disk turn files (read_file).

## Hypotheses (turn 0, blank board)
1. UP is a movement action for a hidden/soon-to-spawn entity (e.g., spawns at bottom edge and
   moves up, or moves an invisible cursor).
2. UP reveals/advances a level step; grid content appears after first action.
3. RESET is a no-op at turn 0 (nothing to reset) — low information value.

## Plan
- Probe with UP. Queue UP,UP (2nd only justified if 1st is benign; queue auto-stops on no-change).
- After each observation: locate any non-zero cells, note their positions and deltas between
  consecutive turns to infer motion direction/speed and spawn rules.
- Watch HUD-like regions (edges) for counters; treat as state, not decoration.

## Log
- t0: blank 64x64. Committing [UP, UP].
