# Draft comment for Kaggle discussion 751598 (NOT POSTED; needs Pranav's go-ahead)

Hi @macruzbar @herbison, adding our team's data points for the rescore review. All statuses below were rechecked via
the Kaggle API on 2026-10-27 UTC.

Same-code control: `pranavagrawai/dfranzen-m2-raw`, scriptVersionId 354278759 (byte-identical copy of Daniel Franzen's
Milestone-2 notebook, no input changes since Oct 1). The same version scored 23.87, 29.44, 24.97, 29.38 and 31.31
between Oct 1 and Oct 17 (3,803-3,888 B outputs).

| Submission | Submitted (UTC) | Notebook / scriptVersionId | Result |
|---|---|---|---|
| 57663110 | 2026-10-26 10:31 | dfranzen-m2-raw / 354278759 (the control above) | COMPLETE, 0.00, 3,411 B |
| 57477659 | 2026-10-21 10:31 | m2-f1-screen2-f1x / 359339222 | COMPLETE, 0.00, 3,411 B; the same version scored 30.88 as 57385881 on Oct 19 |
| 57342990 | 2026-10-18 06:30 | m2-f1-screen1-f1 / 359252919 | COMPLETE, 0.00, 3,411 B |
| 57622267 | 2026-10-25 07:57 | m2-f1x-reap14-screen1 / 360321124 | COMPLETE, 6.52, 3,713 B; partial, its public-25 run of the same version gave 80 levels |

Our own logs show the storage slowdown directly: the notebook's 185 GB model precache took 220-417 s on every run from
Sept 30 to Oct 20, then 2,010 s on Oct 22 and 895 s on Oct 25.

Could engineering include these in the rescore or slot review? Thank you.
