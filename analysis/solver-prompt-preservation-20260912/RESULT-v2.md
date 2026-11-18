# Result: solver-prompt-preservation-local-v2

Status: valid negative; preservation boundary was one rendering stage too early.

Observation:
- The frozen v11 candidate passed admission and traversed the full Actor-to-worker-to-host interaction in 4.88s.
- ACTION1 was delivered after the host refused Actor DONE.
- The exact-equality assertion failed at request 0: model_calls.jsonl contained the interceptor's source messages, while the independently captured HTTP request contained the completion client's cache-control rendering.
- External model calls: 0. ARC environment calls: 0. Spend: $0.

Interpretation:
The candidate preserved all source messages but not the exact provider-bound rendering. NOOA's CompletionClient injects cache-control fields after the interceptor, so capture must apply that deterministic rendering before writing the audit row.

Next action:
Freeze the provider-prepared rendering in model_calls.jsonl and qualify it under a new card. Keep runtime v10 as incumbent until v11 passes its suite.
