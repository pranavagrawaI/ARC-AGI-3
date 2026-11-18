# Procedural transfer rerun 2

Verdict: PASS for feasibility on this single synthetic no-op target.

Qwen 3.8 27B (CoreWeave FP8) submitted UP after two Actor calls and DOWN after two further calls. The first cell inspected the current 3x3 grid; the second recorded upward translation versus no-effect/other-region alternatives with the real set_experiment signature and submitted UP. The host returned the unchanged grid. Call three read last_transition(include_grids=True) and changed_region(). Call four described the state as unchanged and submitted DOWN with rationale: `predict: discriminate center-cell downward translation vs no effect after UP was measured inert`.

This conclusion is narrower than the demonstration's translation conclusion and supported by the actual first transition. The next movement claim remains a prospective hypothesis. No helper/procedure construction, summarizer request, or reported reasoning tokens occurred.

| Measure | Observed |
| --- | --- |
| Calls before action one / between actions | 2 / 2 |
| Received / forwarded / provider responses / worker records | 4 / 4 / 4 / 4 |
| Experiment records | 2 |
| Reported reasoning tokens | 0 on all four responses |
| Provider-reported cost | USD 0.0116112 |
| Cumulative reservation | USD 0.30827512 of USD 1.00 |
| Actor model wall time | 13.0173 seconds |
| Worker/transport errors | none recorded; cleanup returned successfully |

Manifest 5074227fe57be65d92e2a4765347a27514513ea833a3d5e730b1c9d9d52e603e and card hash verified after execution. All four local preflight gates passed on that manifest; see PREFLIGHT.md and the retained preflight directories. Raw provider requests/responses are in raw/, and RESULTS.json preserves the worker records, experiments, and transitions.

Scope: this supports evidence-responsive procedure use on one target within the requested call budget. It does not establish reliability across five transfer cases, a causal benefit of the demonstration, or production gameplay performance. The action names happened to match the example (UP then DOWN); the conclusion changed to measured no-effect. There was no paired live counterfactual run.

Transition caveat: the first UP successor was actually supplied by the synthetic host. The second DOWN submission ends the diagnostic; cleanup records a final unchanged frame without applying the second synthetic action. That terminal accounting row is not evidence about DOWN's effect and was not used to assess adaptation.

Disposition: the compact procedure clears this feasibility gate and may be prepared in a fresh production comparison candidate. No production promotion or further model run was performed in this card.
