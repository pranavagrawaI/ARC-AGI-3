# parity-compat-v13

Question: did V12's three-call deadline, rather than sterile CodeAct, cause Gemma's G3 failure?

Contrast: exact failed V12 Gemma G3 CodeAct arm; only max provider calls changes 3 -> 6. Prior Native call-2 pass is a frozen control and is not rerun.

Measure: earliest calls for capability request, disclosure, post-disclosure computation, evaluator-observed correct answer, and accepted return.

Cap: one CodeAct session, six calls, 16,384 input and 2,048 output tokens/call, $3 maximum; execution disabled until separately authorized.

Stop: accepted return, six calls, integrity/route/provider/budget failure.

Invalid: any packet, prompt, executor, evidence, model route, authority, or prior-control drift.

Actions: pass C4 -> truncation; C5 -> turn-hungry; C6 -> fragile; no pass after observed answer -> policy mismatch; otherwise no compatible completion.
