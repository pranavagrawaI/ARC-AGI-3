# 0159-compaction-recovery-comparison-v1
Question: Does an evidence-linked checkpoint preserve frozen turn-15 orientation better than a mechanical suffix?
Posture: offline frozen-boundary probe.
Contrast: suffix vs checkpoint vs checkpoint+retrieval; same model/probe.
Prediction: checkpoint improves semantic fields; retrieval improves evidence recovery.
Measure: probe-results.jsonl categorical scores, retrievals, raw responses.
Cap: 3 repeats/condition; <= 19 calls; 8192 output tokens/call; 0 ARC actions; 30 min; spend unknown.
Stop: conditions finish, cap, provider failure, or invalid checkpoint.
Invalid if: wrong model/provider, malformed checkpoint, source drift, fallback, or host JSON repair.
Result map: checkpoint gain -> integrate behind flag; no gain -> revise schema; search-only gain -> prioritize retrieval; ambiguous -> add probes.
Outputs: this run directory.
