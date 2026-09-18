# Submission 2026-09-18 — Duck alone (the incumbent), upstream share notebook

Pushed 21:42Z. Kernel `pranavagrawai/taaf-duck-submission-20260918` v1 (commit run RUNNING).
https://www.kaggle.com/code/pranavagrawai/taaf-duck-submission-20260918

## Why this arm
Step 3 paired wave (`runs/step3-paired/RESULT.md`): incumbent 14 levels vs critic 11 over 20
episodes each; precommitted rule → incumbent. Tree `04038a4`; critic package present but
`DUCK_CRITIC_*` unset (0 keys in setup env); actor prompts byte-identical to upstream 7652836.

## Config (option B: upstream shape, safer cap)
| knob | value | upstream `make kaggle-duck` |
|---|---|---|
| concurrency | 28 | 28 |
| per-game cap | 120 min (7200 s) | 132 min |
| experiment envelope | 540 min (soft end only outside a true rerun) | 540 min |
| passes | 1 | 1 |
| rerun estimate | 110 games / 28 = 4 waves × 120 min + ~10 min startup ≈ 8.2 h, ~50 min under the 9 h cap | ≈ 8.97 h |
| commit run | 25 bundled public games offline, 1 wave, ≈ 2.2 h | same |

Rendered `taaf_kaggle_run_share.ipynb` diffs from upstream's published share notebook only
in the source-dataset ref. 28-way on RTX PRO 6000 is upstream's measured regime, not ours
(we measured 10-way: 46–69 s/turn).

## Next
Commit run COMPLETE → `kaggle competitions submit arc-prize-2026-arc-agi-3 -k pranavagrawai/taaf-duck-submission-20260918 -v 1 -f submission.parquet -m "..."`.
One submission per day; none used on 2026-09-18.
