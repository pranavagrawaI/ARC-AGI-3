# solver-prompt-preservation-local-v4
Question: Which exact fields differ between v11's prepared request_messages record and the messages independently observed at the HTTP boundary?
Posture: local diagnosis; no qualification claim
Contrast: recorded model_calls.jsonl request_messages versus fixture-captured http_request_messages.json from the same execution
Prediction: a deterministic field-level diff will identify a post-CompletionClient rendering change, likely cache-control normalization, while the real host path still delivers ACTION1 after DONE refusal.
Measure: one targeted pytest execution followed without rerunning the fixture by a zero-call Python structural diff over the two preserved JSON artifacts in that pytest temporary directory.
Cap: 0 external model calls; 0 ARC environment calls; 1 targeted pytest execution; 1 local structural-diff command; 10 minutes total wall time; spend $0.
Stop: after the first exact differing paths and values are reported; abort on candidate drift, timeout, missing artifacts, or any external request; no retry under this card.
Invalid if: both sides are not preserved from the same execution; the fixture does not reach its equality assertion; the HTTP capture is synthesized from model_calls; or any external model/environment call occurs.
Result map: provider strips only cache metadata → preserve the HTTP-normalized messages at a lower boundary or explicitly label prepared versus transmitted forms; another deterministic transform → reproduce it only if supported by the client contract; no stable diff → keep v10 and instrument the transport request body directly.
Outputs: analysis/solver-prompt-preservation-20260912/DIFF-v4.json and RESULT-v4.md; pytest temporary model_calls.jsonl and http_request_messages.json.
