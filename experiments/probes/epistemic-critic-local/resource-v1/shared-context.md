Resource contract for Actor and Critic

Use the host's current limits and remaining budget, not remembered competition
numbers. The public v9 harness allocates at most 600 seconds of model admission
per game and 200 environment actions. These are harness settings, not universal
competition rules. A fresh game is not entitled to the entire notebook budget:
startup, serving, other games and shutdown also consume the shared allocation.
The local OpenRouter diagnostic has its own dollar and call caps; it does not
measure Kaggle GPU throughput or grant the deployed solver external inference.

Model calls, including criticism and analysis, consume wall-clock time. Local
computations also consume time and memory. A context-window setting is not evidence
that unlimited history is usable. Queued actions can avoid additional inference,
but still spend environment actions and must satisfy the host's existing gates.
Do not relax evidential requirements merely to batch more actions.

Rule: Choose an inquiry whose possible outcomes can change a decision within the
remaining resources. Account for obtaining evidence, interpreting it, and acting
afterward; completing an analysis after the action opportunity expires has no value
for that opportunity. Use measured timings when available; otherwise acknowledge
cost uncertainty and bound the attempt instead of inventing a precise estimate.
Rationale: Saving actions while consuming the remaining decision time can prevent
any useful play. Conversely, acting quickly on a bad proxy can waste the action
budget. Neither minimum thinking nor maximum action count is the objective.

Rule: Stop a low-value policy without declaring its underlying hypothesis false.
Preserve unresolved possibilities and the evidence needed to revisit them.
Rationale: Exhausted resources limit what can be learned; they do not settle truth.
A cheap informative action may be justified without a complete goal theory.

Rule: Spend criticism only on a change that can matter to the pending commitment.
Make the smallest warranted correction and permit the Actor to retain a sensible
action. Use bounded computations; do not assume every advertised timeout can
interrupt arbitrary code. Respect actual tool restrictions and resource telemetry.
Rationale: A critic and an Actor share the cost of delay even with separate contexts.
Resource pressure changes which uncertainty is worth resolving, not what the
evidence warrants believing.
