# Final-observation admission race: local comparison
Question: Does deferring successor publication until host decision admission prevent model work after the final allowed action?
Posture: Diagnosis and narrow correction.
Contrast: Original contingent-controller runtime versus copied submission-runtime-20260910; one-second delayed Agent.cleanup in both arms widens the actual host stop/cleanup window.
Prediction: Original runtime admits an extra model request after action 10; corrected runtime admits exactly 10 requests for 10 actions. Both preserve real official Agent.main and direct/gateway execution.
Measure: requests.json, RESULT.json, worker IPC and subprocess logs; original entrypoint test must expose extra work, corrected suite must pass including exact notebook payload and budget cases.
Cap: One original direct-entrypoint test and one corrected 9-test suite; 180s subprocess limit each; up to 20 controlled requests and 10 executed actions per entrypoint. No GPU or paid model requests.
Stop: Test completion or timeout; no automatic retry or altered fixture.
Invalid if: Wrong runtime, missing evidence, unrelated failure before final-action boundary, orphan worker, or external model call.
Result map: Reproduced original defect and corrected pass permit a new public qualification card; unreproduced or failed correction withholds launch and requires diagnosis.
Outputs: lifecycle-local-original/ and lifecycle-local-corrected/ under analysis/deployment-probes/20260910-first-submission/.
