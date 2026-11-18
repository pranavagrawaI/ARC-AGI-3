# Background-summary removal qualification
Question: Does disabling the background summarizer eliminate summary transport even when reported prompt tokens exceed its trigger?
Posture: Diagnosis and bounded integration qualification.
Contrast: submission-runtime-20260910 versus submission-runtime-20260910-v2, which changes enable_summarizer to False. Both retain host-admitted successor publication.
Prediction: Original direct-entrypoint fixture makes more than 10 model calls for 10 actions at 22000 reported prompt tokens; no-summary runtime makes exactly 10 through both direct and gateway entrypoints. Corrected budget and payload tests pass.
Measure: requests.json, model_calls.jsonl, RESULT.json, subprocess logs; inspect whether extra requests are summarizer requests rather than treating raw request count as useful actions.
Cap: One old-runtime direct-entrypoint test and one new 9-test suite; 180s per subprocess; at most 20 controlled requests and 10 executed actions per entrypoint. No GPU or remote model calls.
Stop: Suite completion or timeout; no automatic retry or fixture adjustment.
Invalid if: Unrelated setup/fixture failure before tested boundary, wrong runtime, missing raw records, orphan worker or external call.
Result map: Triggered old path and corrected pass permit a separately precommitted live public qualification; otherwise diagnose before launching. This does not establish long-run context quality without summarization.
Outputs: summary-local-original/ and summary-local-corrected/ under analysis/deployment-probes/20260910-first-submission/.
