# 0159-compaction-retrieval-comparison-v2
Question: Does a legible deterministic-search interface recover exact older facts omitted by suffix/checkpoint?
Posture: offline frozen-boundary discriminating probe.
Contrast: suffix vs fixed v4 checkpoint vs checkpoint+retrieval with explicit tool semantics; same model/probe.
Prediction: search arm uses tools and scores higher; non-search arms use unknown for absent facts.
Measure: probe-results.jsonl exact 15-field score, tool calls, raw responses.
Cap: 3 repeats/condition; <= 27 calls; 8192 output tokens/call; 0 ARC actions; 30 min; spend unknown.
Stop: conditions finish, cap, or provider/tool failure.
Invalid if: wrong route, checkpoint/source drift, fallback, host repair, or invented IDs accepted.
Result map: search gain -> retain retrieval; no use -> redesign affordance; no gain after use -> revise retrieval/index; ambiguous -> inspect trace.
Outputs: this run directory.
