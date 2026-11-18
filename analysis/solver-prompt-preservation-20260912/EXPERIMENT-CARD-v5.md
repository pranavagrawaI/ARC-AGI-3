# solver-prompt-preservation-local-v5
Question: Does frozen candidate v11 now preserve the exact HTTP-bound Actor messages and every tool-result message while retaining DONE refusal and host action delivery?
Posture: local harness qualification after field-level diagnosis
Contrast: none; local-v4 established that recursive omission of null dictionary fields is the only observed delta between the prepared and HTTP-bound messages.
Prediction: all three Actor records exactly equal independently captured HTTP messages; tool_results equals each request's role=tool subset; the first is empty, the second nonempty; ACTION1 is delivered after DONE refusal.
Measure: one execution of test_done_refusal.py comparing model_calls.jsonl against the fixture HTTP server's captured JSON body and asserting the real host outcome.
Cap: 0 external model calls; 0 ARC environment calls; 1 targeted pytest execution; 10 minutes test wall time; spend $0.
Stop: success on one pass; abort on candidate drift, timeout, assertion failure, or any external request; no retry under this card.
Invalid if: the frozen contract check does not run; HTTP capture is not independent; the path does not reach host action delivery; or any external model/environment call occurs.
Result map: positive → run the unchanged v11 full local suite and freeze check, then report v11 as locally qualified observability successor; negative → retain v10 and move capture to an explicit transport hook; ambiguous → retain v10 and do not replay LS20.
Outputs: analysis/solver-prompt-preservation-20260912/RESULT.md; deployment/kaggle/submission-runtime-20260912-v11; pytest console output quoted in RESULT.md.
