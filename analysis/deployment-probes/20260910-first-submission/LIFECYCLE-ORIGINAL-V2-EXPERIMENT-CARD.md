# Original-runtime race reproduction v2
Question: Does original successor publication admit a model request after the last action during delayed cleanup?
Posture: Diagnosis.
Contrast: Original runtime, compared with already preserved corrected local suite (9 passed in 24.78s).
Prediction: The original reaches 10 executed actions but records more than 10 requests with one-second delayed cleanup.
Measure: requests.json, RESULT.json and subprocess logs in lifecycle-local-original-v2. The previous original attempt failed before execution because its runtime override was relative to worker cwd; it is invalid as a race test and remains preserved.
Cap: One direct-entrypoint local test, 180s subprocess limit, 20 controlled model requests, 10 executed actions; no GPU or external model calls.
Stop: Test completion or timeout, no automatic retry.
Invalid if: Failure before action 10, wrong runtime, missing evidence, orphan worker or external request.
Result map: Extra request reproduces the lifecycle defect and supports qualifying the correction live; no extra request leaves the controlled reproduction inconclusive, with raw Kaggle evidence retained; unrelated failure requires diagnosis.
Outputs: analysis/deployment-probes/20260910-first-submission/lifecycle-local-original-v2/.
