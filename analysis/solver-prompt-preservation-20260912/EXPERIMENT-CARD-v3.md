# solver-prompt-preservation-local-v3
Question: Does frozen candidate v11 preserve the provider-prepared Actor messages, including cache-control rendering and every rendered tool result, while retaining DONE refusal and action delivery?
Posture: local harness qualification after the local-v2 boundary diagnosis
Contrast: none; local-v2 independently showed that pre-render messages differ from the HTTP boundary because CompletionClient injects cache control.
Prediction: all three Actor model-call records exactly equal the HTTP request messages; tool_results exactly equals each request's role=tool subset; the first is empty, the second is nonempty, and ACTION1 is delivered after the host refuses DONE.
Measure: the single pytest result for test_done_refusal.py, whose fixture independently captures JSON request bodies at the HTTP server and compares them with model_calls.jsonl.
Cap: 0 external model calls; 0 ARC environment calls; 1 targeted pytest execution; 10 minutes test wall time; spend $0.
Stop: success on one passing execution; abort on candidate drift, timeout, assertion failure, or any external request; no retry under this card.
Invalid if: the frozen contract check does not run; the HTTP fixture does not independently observe request messages; the path does not reach host action delivery; or any external model/environment call occurs.
Result map: positive → run the unchanged v11 full local suite and freeze check as verification; negative → keep v10 and move capture into the completion-client HTTP boundary; ambiguous → keep v10 and preserve the raw request body in a lower-level fixture before another candidate.
Outputs: analysis/solver-prompt-preservation-20260912/RESULT.md; deployment/kaggle/submission-runtime-20260912-v11 test artifacts; pytest console output quoted in RESULT.md.
