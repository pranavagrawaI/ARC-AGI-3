# 20260920-v12-sb26-su15 result

Kernel `pranavagrawai/tycho-nooa-v12-sb26-su15` v1. ERROR 2026-09-20T19:29Z.
GPU: NVIDIA RTX PRO 6000 Blackwell, 97887 MiB. Not a P100.
Raw: `analysis/deployment-probes/20260920-v12-sb26-su15/kernel-output/`.

## Card classification: INVALID

`result.json` status `invalid`. No `ASSESSMENT.json`. Episode 4 (su15 pass 2) died
`WorkerError: agent exited -6 before submit_actions` (`rc=70`). Worker stderr:
`memory allocation of 1360 bytes failed` then `Fatal Python error: Aborted`
inside CodeAct `execute_code` / pydantic. The notebook used `subprocess.run(...,
check=True)`, so the kernel is ERROR even though three episodes finished.

Do not read this kernel as a 4-episode capability ceiling.

## What still happened (three completed 1800 s walls)

Sequential launcher did what it said: four `# episode` markers, `Game list`
one game each time, no overlapping `started_at`.

| Episode | Stop | Env actions | Wall | L1 |
|---|---|---|---|---|
| sb26 pass 1 | `wall_budget` 1812 s | 87 | 17:54–18:24Z | **yes** — SPACEBAR, env count 85, ~28 min, after a GAME_OVER RESET at 73 |
| su15 pass 1 | `wall_budget` 1808 s | 38 | 18:24–18:55Z | **yes** — CLICK 44 19, env count 23, ~18 min |
| sb26 pass 2 | `wall_budget` 1814 s | 120 | 18:55–19:25Z | no (RESET at 74, still 0) |
| su15 pass 2 | SIGABRT after 6 actions | 6 | 19:25–19:28Z | no, aborted |

`level_0_terminal.json` exists on `sb26-tpzt_evg` and `su15-t_x8gmk8`.
Eviction fired (`EVICTED`); no `EVICTION_ERROR` in any session IPC.
Play continued after both L1s (not the scribe-stall INVALID).

Duck's sb26 L1 is 133–322 s; this L1 was ~1680 s into the episode. su15 L1
here ~1060 s vs Duck 600 s. Cost was enough to finish a level. Tempo is
still far from Duck.

## Reading

The frozen 4-episode card is INVALID. The discriminating *question* still
got a positive on the two completed first-pass episodes: at 1800 s exclusive
wall, v12 completed L1 on both games Duck completes. Pass 2 sb26 did not
repeat it. That is n=3 completed, 2/3 L1, plus a named crash.

Result map applied loosely: do not port a strategy from an INVALID envelope;
do not pretend the L1s did not occur.
