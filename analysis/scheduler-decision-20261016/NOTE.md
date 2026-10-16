# Scheduler workstream: dropped (2026-10-16, Q17 delegated to Claude)

The premise was that the hidden rerun crowds 110 games onto 14 slots (7.9 per slot vs 1.8 public), so allocation
could be a breakthrough. The per-game budget is identical, though. The interactive branch sets
`max_runtime_s_per_game = 532*60*concurrency//110` ("hidden-equivalent share = 7254 s"):

| | games | horizon | tokens/game at 14×53.2 tok/s | slot-hours/game |
|---|---:|---:|---:|---:|
| hidden rerun | 110 | 31,840 s | 216K | 1.126 |
| public screen | 25 | 7,254 s | 216K | 1.128 |

So a scheduler cannot add budget per game. It can only move tokens between games. The 10-13 simulator tested
that class directly (hidden-scheduler-sim-20261013; held-out audit-v1, tail-audit-v1):
- FIFO: −5%.
- Park-after-T: ±1%.
- Retuned token scale: ≈0.
- Hazard index: +16% only if unobserved late solves happen; −32% on recorded trajectories alone, starving ~22 games.

The public→hidden drop (~2x) is then game difficulty, not contention.

Decision: drop the scheduler workstream; GPU goes to friction and mined pieces. It reopens only if notebook
mining turns up an allocation mechanism outside that class, with logs showing it working.
