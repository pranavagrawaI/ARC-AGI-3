# solver-prompt-preservation-local-v6
Question: Does frozen candidate v11 preserve exact HTTP-bound Actor messages and classify both NOOA tool-result renderings, including the first cell's `inspection` PythonOutput, while retaining host action delivery?
Posture: local harness qualification after correcting the tool-result oracle
Contrast: none; local-v5 established exact request equality and showed that results render as both role=tool status messages and role=user PythonOutput blocks.
Prediction: all three request_messages records equal the HTTP fixture; tool_results equals the independent union of role=tool messages and role=user messages containing PythonOutput; request 0 has no call1 result, request 1 contains call1 and `inspection`; ACTION1 is delivered after DONE refusal.
Measure: one execution of test_done_refusal.py over the real Actor-to-worker-to-host fixture and its independently captured HTTP JSON.
Cap: 0 external model calls; 0 ARC environment calls; 1 targeted pytest execution; 10 minutes test wall time; spend $0.
Stop: success on one pass; abort on candidate drift, timeout, assertion failure, or any external request; no retry under this card.
Invalid if: the frozen contract check does not run; HTTP capture is not independent; the path does not reach host action delivery; or any external model/environment call occurs.
Result map: positive → run the unchanged full v11 suite and freeze check; negative → keep v10 and move evidence capture to the transport hook; ambiguous → keep v10 and do not replay LS20.
Outputs: analysis/solver-prompt-preservation-20260912/RESULT.md; deployment/kaggle/submission-runtime-20260912-v11; pytest console output quoted in RESULT.md.
