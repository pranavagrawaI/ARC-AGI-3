# discovery-factorial-20260908-rerun2
Question: With the relay cap fixed and the upstream rate limit reset, can the bounded A/B/C/D OpenRouter screen reach its intended arm comparisons?
Posture: bounded retest; preserve rerun-1 and the original cap failure as evidence.
Contrast: A common working-belief workflow; B measured evidence; C execution-generated discovery demonstration; D both. Same model, tools, reasoning settings, action gates, deadlines, relay cap, and call/spend limits as rerun-1.
Prediction: The reset provider will permit arm A to complete and allow the remaining factors to be compared without changing solver behavior.
Measure: raw requests/responses/reasoning, per-arm RESULTS.json, transitions/actions, relay/provider errors, request byte lengths, and factor-presence checks.
Cap: four arms maximum, one recorded ls20 case per arm, 22 decisions per arm, 48 upstream calls total, 3 cost-reservation units, 1,000,000-byte relay request cap, and 4 hours wall time.
Stop: stop on protocol/factor mismatch, lifecycle failure, unresolved transport retirement, provider error, call/spend cap, or any arm timeout; do not start the next arm after a failed arm.
Invalid if: any factor differs from its assigned arm, raw provenance is incomplete, the candidate source drifts, or a response is interpreted without an attributable request.
Result map: all required arms complete -> compare outcomes and advance only a repeatable practical winner; provider still rate-limits -> preserve evidence and stop provider-dependent screening; partial/ambiguous -> preserve evidence and issue a new card before another change.
Outputs: analysis/deployment-probes/20260908-discovery-factorial-rerun2/raw/openrouter/ and the unchanged candidate source under deployment/kaggle/designer-player-discovery-20260908/.
