# solver-prompt-preservation-local-v2
Question: After freezing candidate v11, does one targeted real-path execution preserve the exact Actor request messages and rendered tool-result messages in model_calls.jsonl while retaining DONE refusal and action delivery?
Posture: local harness qualification; retry of invalid local-v1 after candidate-freeze repair
Contrast: none; local-v1 already established the copied v10 schema lacks the fields, while this run qualifies the frozen v11 behavior.
Prediction: all three Actor records exactly equal the independently captured HTTP request messages; each tool_results value equals that request's role=tool subset; the first is empty, the second is nonempty, and ACTION1 is delivered after the host refuses Actor DONE.
Measure: pytest output for test_done_refusal.py and its assertions over the model_calls.jsonl rows produced by the real Actor-to-worker-to-host fixture.
Cap: 0 external model calls; 0 ARC environment calls; 1 targeted pytest execution; 10 minutes test wall time; spend $0.
Stop: success on one passing targeted execution; abort on any external request, candidate drift, timeout, or assertion failure; no retry under this card.
Invalid if: the frozen contract check does not run; the fixture HTTP server does not independently capture the request body; the path does not reach host action delivery; or any external model/environment call occurs.
Result map: positive → run the unchanged v11 full local suite under a separate verification step and retain v10 as gameplay incumbent until a separately authorized live run; negative → keep v10 and diagnose only the failed preservation boundary; ambiguous → keep v10 and write a narrower serializer probe.
Outputs: analysis/solver-prompt-preservation-20260912/RESULT.md; deployment/kaggle/submission-runtime-20260912-v11 test artifacts; pytest console output quoted in RESULT.md.
