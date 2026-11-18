# first-submission-entrypoint-20260910-v3
Question: Does the shared entrypoint complete both direct-public and HTTP gateway modes when the controlled responder varies actions instead of repeating a known no-impact action?
Posture: Local integration qualification; v2 hit the real same-state no-impact rejection after seven UP actions, then the refusal-deadlock guard. Preserve that negative control.
Contrast: none; controlled UP/DOWN/LEFT/RIGHT rotation, not model intelligence.
Prediction: Both surfaces execute ten Solver actions and reconcile 10–20 controlled requests, then close workers and scorecards.
Measure: entrypoint-v3/ requests.json, RESULT.json, recorded frames, IPC transitions, worker metadata and pytest checks.
Cap: Two LS20 sessions, ten Solver actions and twenty controlled requests per session, 180s outer subprocess limit each; 1200s nominal scheduler budget; zero paid calls/GPU quota.
Stop: Ten actions per surface or failure; preserve failure and clean up local model/ARC servers.
Invalid if: Source drift, external model/game traffic, lost accounting, timeout or failed worker completion.
Result map: Pass -> bounded Kaggle public test of unchanged submission notebook; failure -> diagnose and create a fresh attempt/card. No solver-quality claim.
Outputs: analysis/deployment-probes/20260910-first-submission/entrypoint-v3/; earlier attempts preserved.
