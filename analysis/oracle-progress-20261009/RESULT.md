# Oracle progress probe (2026-10-09)

Scripts: `replay.py` (local deterministic replay), `bfs_probe.py`, `oracle.py build|score`, `analyze.py`. Graphs in `graphs/`, per run x level distances in `rows.jsonl`, build log in `build.log`.
Engine: the `solvers/tycho_nooa/.venv` python (arcengine), with game files from `arc-agi-3-reference/competition-env/environment_files`.

## Findings
- **Replay is deterministic:** all 25/25 games of dfz-control reproduce their logged levels when the logged actions are replayed locally. Any visited state can be rebuilt offline.
  This also makes resuming from a checkpoint possible.
- **Local stepping runs at about 5,300 steps/s.** Deepcopy takes 7 ms. A game copy is 0.2-1.2 MB, so the BFS frontier stores action paths instead.
- **Exhaustive BFS in 240 s:** 6 of 75 game-levels were exact, and 11 more had a shortest path found. 58 stayed open, almost all beyond L0.
  Component-click candidates and path replay make expansion slow (about 2k states per 240 s). Scaling needs an abstract state hash from the game source, or cheaper click candidates.
- On those 15-17 mostly-L0 levels:
  - opt_rate (share of moves that reduce the true distance by 1) has Spearman 1.0 with hidden: 0.50 / 0.51 / 0.54 / 0.64 / 0.65 for Duck3.6 / Q3.8 / KF / P1 / dfz.
    But it puts P1 next to dfz, while their hidden scores differ 8x. It measures decision quality per action, not the rate.
  - prog_per_min has Spearman 1.0 and resid sd 0.37.
  - Weak systems earn partial credit on stuck levels (0.2-0.3 of the distance), so the signal is dense.
  - Noise: identical reruns differ in opt_rate by +0.09 to +0.10 (t 1.2-1.8) at n = 12-13 levels. Too few levels to hill-climb on yet.
