# Submission receipt — ref 56355127

Submitted 2026-09-19 09:37:11Z: kernel `pranavagrawai/taaf-duck-submission-20260918` v1,
status PENDING at submit. Message: "Duck alone (upstream 7652836 prompts, tree 04038a4),
28-way, 120 min/game; incumbent from Step 3 paired wave".

## Commit run (the qualification run for these exact bytes)
Kernel RUNNING 21:42Z → COMPLETE ~23:52Z 2026-09-18 (2 h 10 min): ~9.5 min startup, then all
25 bundled public games to the 7200 s cap in one 28-way wave. No errors; teardown ran;
dummy `submission.parquet` written. Output in `kaggle-output/`.

| metric | value |
|---|---|
| games with ≥1 level | 13 / 25 |
| total levels | 17 (ar25 2, re86 2, tu93 2, vc33 2; bp35, cn04, lf52, lp85, ls20, r11l, sb26, sp80, su15 1) |
| TAAF mean score | 1.90 |
| tokens per game at cap | ~57–59k generated |

## Rerun projection
110 games / 28 = 4 waves × 120 min + ~10 min startup + gateway wait ≈ 8.2 h against the 9 h
cap. The commit run's single wave took 130 min end to end, consistent with the projection.
Reference failure: submission 56303234 (2026-09-17) ERRORed at ~51 min, in startup.

## Watch
`kaggle competitions submissions arc-prize-2026-arc-agi-3` — PENDING → COMPLETE/ERROR.
The rerun log is visible only in the Kaggle UI.

## Outcome (read 2026-09-20)

**COMPLETE, public score 0.67** (RHAE %, averaged over the 55 hidden-public games; the other 55
score the private board). Leaderboard snapshot 2026-09-20: rank 1314 / 3166 teams; median 0.29,
p75 1.71, top 18.81 (Tufa Labs, the harness authors). The rerun ran its full length (PENDING from
09:37Z through at least 13:38Z), so the 9 h envelope and 28-way × 120 min pacing held on 110 games.

### Why 0.67 and not ~2
`rhae/estimate.py` scores the commit run's 25 public games with the human baselines from the
competition's `environment_files/*/metadata.json`: **2.02 %**. The same levels at perfect human
efficiency would be **2.55 %**. So on the games we can see, action efficiency costs ~0.5 pp and
level *depth* costs the rest: completing L1 of an 8-level game is worth at most 1/36 of that game
whatever the efficiency, and Duck's deepest game reached L2 of 8. The hidden-public games came in
at 0.67 % — a 3× drop from the visible 25 — which is a completion-rate gap on unseen games, not
an efficiency gap. Efficiency tuning cannot reach 1 %; only reaching L2/L3 on many games can.

The critic's stated lever (unstick, go deeper) is the right lever for this metric; Step 3 says
this critic does not pull it. Recorded, not actioned.
