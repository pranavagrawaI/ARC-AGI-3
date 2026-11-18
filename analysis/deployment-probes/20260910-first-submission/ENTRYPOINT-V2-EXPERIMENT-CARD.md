# first-submission-entrypoint-20260910-v2
Question: Does the shared entrypoint execute and account for ten local actions in both direct-public and gateway-HTTP modes?
Posture: Qualification after v1 test budget mismatch: its 300s nominal budget left only a 1s game admission after reserves and produced one action.
Contrast: none; two integration surfaces with controlled model replies.
Prediction: Ten requests and ten Solver actions per mode; payload hashes match; successful scorecard close and worker completion.
Measure: entrypoint-v2/pytest output, requests.json, RESULT.json, stdout/stderr and worker IPC.
Cap: Two local LS20 sessions, ten Solver actions and twenty loopback requests each, 180s outer timeout per session. Inner scheduling budget 1200s to test normal allocation; outer bound is 180s. Zero paid calls or Kaggle quota.
Stop: Ten Solver actions or integration failure; clean up controlled model and local ARC HTTP server.
Invalid if: Lost accounting, source drift, external model/game traffic, timeout, failed cleanup or incorrect gateway branch.
Result map: Pass -> bounded Kaggle public qualification. Failure -> preserve evidence and repair before a new run/card. Gateway mock uses real ARC engine over HTTP but is not Kaggle sidecar evidence.
Outputs: analysis/deployment-probes/20260910-first-submission/entrypoint-v2/; v1 remains preserved.
