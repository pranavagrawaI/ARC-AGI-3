# Submission entrypoint v4 — all-game budget repair
Question: Does the corrected wrapper retain local and gateway execution while allocating positive bounded admission for 110 games?
Posture: Qualification, not solver-quality evaluation.
Contrast: none.
Prediction: Exact embedded payload test and two controlled-response real-game entrypoint tests pass; budget checks pass for 1, 3, 25, 55, 110, 135 games.
Measure: pytest results, preserved subprocess logs, RESULT.json, requests.json and worker IPC; budget arithmetic includes 120s final decision and 30s cleanup per game plus 180s final reserve.
Cap: One local suite, 180s per subprocess, at most 20 controlled model requests and 10 actions per entrypoint; no remote model requests or GPU run. Monetary spend not applicable.
Stop: Suite termination; no automatic fixture changes or retries.
Invalid if: Missing evidence, unexpected external request, changed donor package, orphan worker, or timeout.
Result map: Pass permits a separately precommitted public v2 qualification; failure requires diagnosis and a new card; neither licenses competition submission.
Outputs: analysis/deployment-probes/20260910-first-submission/entrypoint-v4/.
