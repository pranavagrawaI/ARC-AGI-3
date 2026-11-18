# discovery-factorial-20260908-rerun3
Question: Does declarative Context preserve the measured-evidence factor through context compression, allowing the bounded A/B/C/D screen to complete?
Posture: bounded repair/retest; preserve all prior runs.
Contrast: A baseline; B measured evidence; C execution-generated demonstration; D both. Same model, tools, reasoning, gates, deadlines, relay cap, and call/spend limits as rerun-2. Only the context registration fix and run identity differ.
Prediction: B and D retain the evidence block after summarizer/context rebuilds; A remains unchanged; the screen reaches later arms unless the provider fails independently.
Measure: raw requests/responses/reasoning, factor-presence checks per request, per-arm RESULTS.json, transitions/actions, and lifecycle/provider errors.
Cap: four arms maximum, one recorded ls20 case per arm, 22 decisions per arm, 48 upstream calls total, 3 cost-reservation units, 1,000,000-byte relay request cap, 4 hours wall time.
Stop: stop on factor mismatch, lifecycle failure, unresolved transport retirement, provider error, call/spend cap, or arm timeout; do not start another arm after a failed arm.
Invalid if: source manifest drifts, factor presence is not checked per request, or request/result provenance is incomplete.
Result map: all arms complete -> compare candidates; evidence survives but provider stops -> retain partial evidence and diagnose provider; evidence still disappears -> diagnose context registration before any quality claim.
Outputs: analysis/deployment-probes/20260908-discovery-factorial-rerun3/raw/openrouter/.
