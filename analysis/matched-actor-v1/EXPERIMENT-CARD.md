# matched-actor-v1
Question: with model, evidence, computation, authority, schema, and ceilings fixed, which control language yields better decisions per resource?
Contrast: Native typed tools + fresh value-only Python versus CodeAct proxy + persistent Python.
Prediction: Native leads A; CodeAct earns its tax only by improving C.
Measure: report.json correctness, calls, tokens, cost, latency, cells, disclosures, invalid/repair interactions, milestone calls.
Cap: 24 sessions; 8 calls/session; 192 calls; 4096 output tokens/call; 120 s/session; $3; 0 ARC actions.
Stop: completion, any cap, route/provider failure, or integrity violation.
Invalid if: committed drift, unequal evidence/computation/authority, key disclosure, missing trace, route drift.
Result map: Native correct+efficient -> Native; CodeAct C gain at tolerable cost -> isolate mechanism; equal accuracy with CodeAct cost -> Native; seam-local failure -> replicate before redesign.
Outputs: analysis/matched-actor-v1/runs/<run-id>/
