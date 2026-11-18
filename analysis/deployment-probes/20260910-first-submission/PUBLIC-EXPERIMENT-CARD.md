# first-submission-public-20260910-v1
Question: Does the locally qualified submission notebook run the frozen 27B FP8 Solver on three public games on Kaggle and shut down cleanly?
Posture: Competition submission readiness; no improvement claim.
Contrast: none; sequential LS20/S5I5/CD82, same frozen Solver as yesterday's contingent v3, reasoning off and MTP off.
Prediction: Model/runtime setup succeeds; all three games execute Solver actions with authoritative successors; normal per-game cap/admission/win stopping and owned-server cleanup complete; the notebook emits the commit-mode parquet.
Measure: qualification/result.json, official-launcher-result.json, game-progress.jsonl, request-count-facts.json, server.log, teardown classification, lifecycle records and per-session IPC. Compare BUILD.json and embedded CHASSIS identity.
Cap: One private Kaggle RTX PRO 6000 run; 7200-second platform limit (2 GPU-hours maximum); three public games, ten Solver actions maximum each; 1200s game admission maximum, 300s worker decision limit, 180s request timeout, 16384 output tokens/request, 4800s launcher and 6300s outer subprocess timeout. Zero automatic relaunches; zero competition submissions under this card. Monetary spend not applicable to quota.
Stop: Complete all three games and cleanup. Abort on setup/worker/HTTP error, timeout, failed identity/accounting or cleanup; no server restart. Notebook can stop an individual game at its predeclared action/admission/win boundary.
Invalid if: Wrong candidate/model/runtime/GPU, OOM, lost request/action accounting, dropped game, uncontained process/port, timeout or missing terminal evidence. A successful dummy parquet is not sufficient.
Result map: Pass -> submit this exact successful notebook version once under a separate competition-run card. Failure -> preserve raw outputs, diagnose the boundary, fix under a fresh candidate/version/card before launch. No solver-performance inference.
Outputs: analysis/deployment-probes/20260910-first-submission/kaggle-public-v1/ (new directory on retrieval); kernel pranavagrawai/arc3-first-submission-20260910. BUILD.json sha256 identifies code at launch.
