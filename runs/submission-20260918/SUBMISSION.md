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
