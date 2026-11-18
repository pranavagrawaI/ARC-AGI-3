# Result: solver-prompt-preservation-local-v5

Status: valid partial positive; exact request preservation passed, but the dedicated tool-results oracle was incomplete.

Observation:
- Frozen v11 traversed the real path and delivered ACTION1 after DONE refusal in 4.96s.
- Exact equality between all three model_calls request_messages values and all three independently captured HTTP message lists passed.
- Equality between the recorded tool_results and the test's role=tool subset also passed.
- The final assertion expecting no tool result in request 0 failed: NOOA includes prefill tool-status messages before the first Actor call.
- Inspection of the preserved HTTP artifact showed the actual cell stdout (`inspection`) in a role=user `<sys ...>PythonOutput(tool_call_id='call1'...)` message, not a role=tool message.
- External model calls: 0. ARC environment calls: 0. Spend: $0.

Interpretation:
The main request_messages field is now exact and already preserves the missing cell output. The separate tool_results field must include both API role=tool messages and NOOA's rendered PythonOutput user-message blocks; the prior oracle classified only the first representation.

Next action:
Expand the dedicated extraction and independent test oracle to both rendered forms, freeze v11, and run one final targeted qualification under a new card.
