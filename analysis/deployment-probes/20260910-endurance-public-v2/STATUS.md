# Public endurance v2

Launched Kaggle version 6 after explicit user authorization to fix and relaunch.
Terminal status: COMPLETE. Audited operational PASS under the frozen card;
zero levels completed. See RESULT.md and AUDIT.json. All 178 output files retained.
No competition submission.

Only runtime-affecting change from v1: notebook startup timeout 600 -> 1200,
compatible with the unchanged r8-l3b parser. Solver/runtime, play.py, public
games, 600-second admission windows, 200-action caps and absolute notebook
deadlines are unchanged.

New exact packaged launch-command regression failed before and passed after
the fix. 22 targeted tests passed; 12 unchanged real-entrypoint cases were not
rerun. Frozen build files and downloaded remote notebook source match.

Evidence: regression-before.log, regression-after.log, push.log,
prelaunch-quota.log, prelaunch-status.log, source/, runtime-source/,
remote-source/, EXPERIMENT-CARD.md. Failed v1 remains untouched.

Terminal outputs and raw ledgers verified. No retry launched. Public endurance
is qualified for this run; reliable solving and nine-hour endurance remain unproven.
