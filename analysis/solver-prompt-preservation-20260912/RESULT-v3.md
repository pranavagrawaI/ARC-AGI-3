# Result: solver-prompt-preservation-local-v3

Status: valid negative; applying NOOA's explicit cache-control renderer did not reproduce the HTTP-bound messages exactly.

Observation:
- Frozen candidate v11 traversed the real path and delivered ACTION1 after DONE refusal in 4.80s.
- Exact equality still failed at request 0 after `_inject_cache_control` was applied to the recorded messages.
- External model calls: 0. ARC environment calls: 0. Spend: $0.

Interpretation:
Another transformation exists between CompletionClient's prepared messages and the JSON body observed by the HTTP fixture, or the OpenAI client strips the injected field. The assertion output is too abbreviated to identify the field safely.

Next action:
Persist the independently observed fixture request messages as a raw diagnostic artifact in one new run, diff them programmatically against model_calls.jsonl, and move capture to the proven boundary if necessary.
