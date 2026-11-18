# discovery-factorial-20260908-rerun1
Question: After removing the relay's premature 120,000-byte input cap, can the same bounded A/B/C/D OpenRouter screen reach its intended arm comparisons?
Posture: bounded repair/retest; preserve the failed first run as evidence.
Contrast: A common working-belief workflow; B measured evidence; C execution-generated discovery demonstration; D both. Same model, tools, reasoning settings, action gates, deadlines, and call/spend limits as the first card.
Prediction: The repaired relay will admit valid growing prompts without changing solver behavior; at least A will reach the action path.
Measure: raw requests/responses/reasoning, per-arm RESULTS.json, transitions/actions, relay errors, request byte lengths, and factor-presence checks.
Cap: four arms maximum, one recorded ls20 case per arm, 22 decisions per arm, 48 upstream calls total, 3 cost-reservation units, 1,000,000-byte relay request cap, and 4 hours wall time.
Stop: stop on protocol/factor mismatch, lifecycle failure, unresolved transport retirement, provider error, call/spend cap, or any arm timeout; do not start the next arm after a failed arm.
Invalid if: any factor differs from its assigned arm, raw provenance is incomplete, the candidate source drifts, or a response is interpreted without an attributable request.
Result map: all required arms complete -> compare outcomes and advance only a repeatable practical winner; repaired relay still fails -> diagnose the next bounded transport/context limit; partial/ambiguous -> preserve evidence and issue a new card before another change.
Outputs: analysis/deployment-probes/20260908-discovery-factorial-rerun1/raw/openrouter/ and the unchanged candidate source under deployment/kaggle/designer-player-discovery-20260908/.
