# first-submission-entrypoint-20260910-v1
Question: Does the new submission entrypoint execute the frozen solver on the local real game and preserve notebook payload identity?
Posture: Local qualification before Kaggle upload.
Contrast: none; deterministic loopback replies, no live-model claim.
Prediction: Ten controlled requests produce ten official-loop Solver actions on LS20 and complete request accounting; embedded candidate hashes match.
Measure: entrypoint-v1/pytest output, requests.json, RESULT.json, worker IPC and notebook payload assertions.
Cap: One local LS20 game, ten Solver actions, twenty loopback requests, 180-second subprocess timeout, zero model spend/GPU quota.
Stop: Finish both tests; stop game at ten actions; abort on timeout, identity drift or integration failure.
Invalid if: Lost request/action accounting, wrong frozen candidate, external model call, timeout, failed worker completion.
Result map: Pass -> bounded Kaggle public run of this notebook; failure -> preserve trace, repair with a new attempt/card. This does not qualify gateway mode or model quality.
Outputs: analysis/deployment-probes/20260910-first-submission/entrypoint-v1/.
