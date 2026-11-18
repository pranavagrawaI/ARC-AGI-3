# First submission — 10 September 2026

## Current state

Final checkpoint for this turn: remote version 3 ended ERROR. LS20 completed 2 actions; S5I5 completed nine model calls but no action before the 120s decision timeout; CD82 was not reached. No summaries ran. Resources closed but the interrupted model request invalidates qualification. All raw output is archived in kaggle-public-v4/. No competition submission made; 8.88 GPU hours remain. No further GPU launch is scheduled pending the coverage/depth policy choice. Read REPORT.md for the consolidated handoff. Earlier checkpoints below are historical, not current verdicts.

Latest: remote version 2 completed but is withheld: whole server.log shows two abandoned/deleted request-state events at earlier game boundaries despite clean final teardown. See PUBLIC-V3-RESULT.md. Local source v4 disables the background summarizer while retaining host-admitted successors. At 22000 reported prompt tokens, original configuration produced 10 Actor plus 7 summary calls; corrected direct/gateway each produced 10 Actor calls and no summaries. Nine tests passed in 25.50s. PUBLIC-V4-EXPERIMENT-CARD.md governs the newly pushed candidate (expected remote version 3); notebook SHA256 3c03eb71a7f461bd95705c1b7f2d19ad5d628c2103accc4bff4aa6c24a3c9728, runtime contract 8fa8984d934ca393c35d2bb5692e54e7c8004e1c12c0be0a918f57685a42406d. No competition submission yet.

### Prior checkpoints

Public v1 ended ERROR after 30 executed actions and 205 completed model calls. Final teardown had in-flight model work: operational qualification failed. Full raw output is archived in kaggle-public-v1/. See PUBLIC-V1-RESULT.md.

Corrected candidate submission-20260910-v3 was pushed successfully as Kaggle version 2. Its one-hour PUBLIC-V3-EXPERIMENT-CARD.md is active. Notebook SHA256 345abfcd9407f5b8500975d31ce52e9c9cb362baa3a3197f0c9f9f8e14473930. New runtime contract de91c8dd7097a7cdc40ef8332a97f362d9251837f9830b4a6b4d6b4bbb74640c. Besides the all-game budget repair, successor publication now requires another host decision admission. Original delayed-cleanup reproduction made 11 requests for 10 actions; corrected direct/gateway tests each made exactly 10. Corrected 9-test suite passed. Budget-only v2 source was never uploaded.

No competition submission has been made. Public v2/v3 source naming is local history; remote notebook version 2 is the lifecycle-corrected candidate. Do not confuse them.

## Earlier checkpoint (superseded by current state above)

Competition submission not yet made. API reports numAllowedNow=1.

Local direct smoke passed (3 controlled responses/actions). Entrypoint v1 and v2 fixture failures are retained; v3 passed 3 tests, including 10 actions through each of the direct and simulated HTTP gateway entrypoints. These are real game/host tests with controlled model responses, not live local 27B inference.

Kaggle public v1 is running: pranavagrawai/arc3-first-submission-20260910, version 1. Notebook SHA256 9dcf6b716fbe78cbe6854262ff9a8ef55b9742fa8711d52d625f01d8e738ed7d. Its PUBLIC-EXPERIMENT-CARD.md remains frozen.

Version 1 is withheld from competition irrespective of its public result: its admission formula (timeout-180)/game_count-330 is negative at 110 supplied games. Public three-game qualification cannot detect this defect. The original source directory is preserved unchanged. Corrected source is deployment/kaggle/submission-20260910-v2; entrypoint-v4 qualifies the repair before any second GPU run.

Corrected policy: maximum 75s admission/game, 120s worker decision timeout, 90s client request timeout. Reserve final decision plus 30s cleanup for every supplied game and 180s final cleanup. Budget tests include 110 and 135 games. Gateway enumeration now retains all full game IDs instead of collapsing versions by prefix. This policy prioritizes bounded full-set execution; dependable solving and sufficient depth remain unproven.

Installed Kaggle CLI kernels_output parses a version suffix but does not pass it in ApiListKernelSessionOutputRequest. Do not rely on suffix to identify evidence; retrieve terminal outputs before pushing the next version, into a fresh directory and verify source/run identity. live-snapshot-01 returned no log while running.
