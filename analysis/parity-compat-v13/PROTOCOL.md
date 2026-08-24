# V13 CodeAct runway diagnostic

V13 reuses the exact committed V12 `g3-composition` gate and Gemma route. It runs CodeAct only. The earlier Native G3 pass on provider call 2 is committed as the control and is not sampled again.

The sole treatment change is the controller ceiling from three to six provider calls. That ceiling is absent from model-visible prompts, manifests, and provenance, so this is truncation removal rather than a disclosed-policy change.

T1 records the earliest provider call for capability request, evidence disclosure, a completed post-disclosure Python cell, an evaluator-observable correct `(L0:T1, 11->5)` pair, and accepted `return_result`.

Interpretation is precommitted: call 4 means deadline truncation; call 5 means compatible but turn-hungry; call 6 means compatible but fragile; failure after the correct answer becomes observable means interface-policy mismatch; otherwise no compatible completion. No Native rerun, API change, decoder, raw representation, extra disclosure, ARC action, Builder, goal, planner, or theory authority is permitted.
