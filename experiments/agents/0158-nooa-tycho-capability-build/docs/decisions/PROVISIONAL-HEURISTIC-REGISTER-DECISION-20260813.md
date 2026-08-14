# Provisional heuristic register decision — 2026-08-13

## Correction to the v25 frame

V25 treated visible change outside the claimed primary region as a residual to be classified `explained`, `auxiliary`, `unresolved`, or `none`. That frame was too totalizing. In LS20, the bottom strip advances after every action because a step elapsed; it can change independently of action identity. Requiring the chosen-action hypothesis to close every visible change would force independent environment dynamics into that hypothesis.

The v25 run also showed that semantic closure labels were self-certifying: Qwen called 42 outside changes “explained” without making that interpretation prospectively load-bearing. Its seventh claim then used `none` despite mechanically visible outside change.

## Candidate change

All residual-closure labels are deleted. Mechanical inside/outside counts remain immutable observations, not explanatory debts. Every post-probe claim must instead include:

- a probe-bound `retain`, `revise`, or `abandon` update;
- a full register of 1–12 model-authored heuristics;
- for each heuristic, a stable name, belief, explicit scope, evidence references, decision role (`active`, `watch`, or `defer`), and a condition for reconsideration;
- host-inserted `epistemic_status: provisional`.

Existing heuristic names and their evidence references may accumulate or be revised but cannot silently disappear. No API state represents explained, confirmed, closed, or certain. Deferral is an operational attention choice, not an epistemic conclusion.

## Delegation boundary

The host validates syntax, probe binding, evidence existence, monotonic preservation, and the fixed provisional status. It continues to report transition-local effect matches and mechanical visible-change counts. It does not group changes, decide whether two observations share an identity, infer causal independence, or choose which heuristic should guide action.

The model authors all interpretations and attention roles. A bottom-strip regularity can therefore be watched under a stated tested scope while a directional-motion heuristic actively guides action. Either remains revisable when its stated reconsideration condition occurs.

## Qualification boundary

Offline tests show that a heuristic can be actively used, watched, or deferred while remaining provisional; `confirmed` is rejected; prior heuristic names and evidence cannot disappear; and the real sandbox rejects those omissions before protected-host parsing so CodeAct can repair them. The historical action-map shadow still reaches G2.

This does not establish that Qwen will maintain a useful register, separate action identity from elapsed-step dynamics, learn LS20 directions, or complete a level. Those are fresh-run questions.
