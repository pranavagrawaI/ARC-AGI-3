# First-submission preparation — 10 September 2026

Not yet competition-ready. No competition submission has been made. The account has 8.88 GPU hours remaining; these three public attempts used approximately 1.12 GPU hours. Last submission-limit readback allowed one attempt.

## Established

The current Solver can execute through the real official Agent loop and an HTTP ARC gateway locally using controlled model responses. The laptop's 8GB GPU cannot run the intended 27B FP8 configuration. Live model qualification therefore ran on Kaggle.

| Remote version | Local candidate | Observed result | Decision |
| --- | --- | --- | --- |
| 1 | submission-20260910 | 10 actions each on LS20/S5I5/CD82, 0 levels, 205 completed model calls; active request at final server teardown | Failed; withheld |
| 2 | submission-20260910-v3 | 2/4/1 actions, 0 levels, 54 completed calls; final teardown clean but two abandoned request-state notices at earlier game boundaries | Notebook complete, but withheld after whole-log review |
| 3 | submission-20260910-v4 | LS20 2 actions; S5I5 9 completed model calls before its first action, then 120s decision timeout; CD82 not reached | Failed; withheld |

Every raw run was downloaded before the next upload, into separate kaggle-public-v1/v3/v4 directories. Model/runtime identity was checked; latest runtime files match contract 8fa8984d934ca393c35d2bb5692e54e7c8004e1c12c0be0a918f57685a42406d. Latest containment scans show resources closed; server.log has one deleted-state notice following the timeout and no out-of-memory text markers. Clean resource containment does not make its interrupted request valid.

## Changes and verification

The obsolete canonical starter shell was not silently promoted. Dedicated submission candidates package the actual Solver/model/runtime, use sequential official Agent.main, and select public offline or competition gateway mode. Local budget tests cover up to 135 supplied games; the initial wrapper's negative budget at 110 games was fixed before any competition submission.

Successor observation publication originally woke the Actor before the host decided whether to stop. A copied runtime now defers publication until another host decision is admitted, while preserving authoritative transitions immediately. A delayed-cleanup reproduction made 11 requests for 10 actions before the fix and 10 afterward.

NOOA background summarization could independently outlive a game. The latest short-session candidate disables it. A schema-valid 22000-reported-token local comparison produced 10 Actor +7 summary calls in the prior configuration versus 10 Actor +0 summary calls in both corrected entrypoints. The latest suite passed 9 tests in 25.50s. Failed earlier fixtures are preserved, including relative-path and mock JSON-schema errors; those are not solver-quality evidence.

## Remaining decision

Latest source: deployment/kaggle/submission-20260910-v4; latest runtime: deployment/kaggle/submission-runtime-20260910-v2. The notebook is remote version 3 and failed public qualification. Do not submit it.

The 120-second worker decision cap is not a demonstrated operating envelope: S5I5 used it on nine analysis/tool-generation calls without an action. Increasing every game's reserve blindly conflicts with the full-set budget. Allowing graceful per-game budget exhaustion, including games with zero Solver actions, would be a new explicit competition policy requiring implementation, cancellation/retirement accounting, and fresh public qualification. Alternatively, prioritize longer or more efficient solving before submission. No further GPU launch is scheduled pending this meaningful coverage/depth choice.

Separately unresolved: installed Kaggle CLI describes push timeout as session-specific, but inspected docs do not explicitly establish inheritance behavior for competition reruns. Check or conservatively avoid relying on a short commit-session timeout for a longer competition run. Explicit-version output downloads ignore the suffix internally; version-suffixed source pull returned 403. Latest source readback matched remote version 2 while it was current. Preserve immutable identity checks around any eventual submission.

The next competition run, once qualified, must have its own precommitted card and submit the exact successful remote notebook version. A dummy parquet, a green status, and completed-call counts alone are insufficient.
