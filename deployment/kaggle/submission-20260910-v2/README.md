# First submission candidate — 10 September

`submission.ipynb` packages the exact live-v3 contingent-controller runtime
(`2c1e13ef4ce25e59121bcd6c637a869a8cb09c49fc58712fc9212234b8c262c9`),
27B FP8, reasoning off, MTP off. `BUILD.json` identifies notebook and entrypoint.
The original starter notebook remains untouched; use this candidate's metadata.

Commit mode runs LS20/S5I5/CD82 sequentially, up to ten Solver actions each.
Competition rerun mode discovers the supplied gateway games and plays them
sequentially under per-game admission budgets. Both use the same official
Agent loop, adapter and frozen NOOA worker. Sequential scheduling is an explicit
submission integration choice, not a measured performance improvement.

Local `entrypoint-v4` qualification passed (9 tests): budgets through 135 games, embedded hashes, ten real LS20
actions via direct offline mode and ten via a local ARC HTTP gateway, controlled
model replies and exact accounting. Earlier attempts are retained in
`analysis/deployment-probes/20260910-first-submission/`; one exposed an undersized
test scheduling budget and one the fixture's repeated known-no-impact action.
These tests establish integration, not Qwen quality or Kaggle sidecar parity.

The private public-game run must complete with correct identity, requests,
actions and cleanup before this exact version is submitted. Dummy parquet is
emitted only after successful public qualification, following the official
starter contract; the competition gateway generates the actual rerun output.

Admission is capped at 75 seconds per game, worker decisions at 120 seconds,
and client model requests at 90 seconds. The 110-game budget reserves the final
decision and cleanup separately from admission. This bounds an initial full-set
run; it does not establish sufficient depth for solving.

Build: `python3 deployment/kaggle/submission-20260910-v2/build.py`.
Do not rebuild after launch and call it the same candidate. Preserve the pushed
version and all failed raw outputs in separate directories.
