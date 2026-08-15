# 0159-compaction-retrieval-comparison-v3
Question: Do decision-shaped retrieval views fix the v6 motion and recovery-request errors efficiently?
Posture: offline frozen-boundary discriminating probe.
Contrast: suffix vs fixed v4 checkpoint vs checkpoint+structured retrieval; same model/probe.
Prediction: search reaches 15/15 and uses fewer prompt tokens than v6; non-search arms abstain.
Measure: probe-results.jsonl exact 15-field score, tool calls, raw responses.
Cap: 3 repeats/condition; <= 21 calls; 8192 output tokens/call; 0 ARC actions; 30 min; spend unknown.
Stop: conditions finish, cap, or provider/tool failure.
Invalid if: wrong route, checkpoint/source drift, fallback, host repair, or invented IDs accepted.
Result map: 15/15 cheaper -> promote views; residual error -> revise the failing view; no gain -> reject this representation.
Outputs: this run directory.
