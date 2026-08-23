# parity-compat-v4
Question: Does congruent structured action history fix Kimi K3's post-query CodeAct transport failure?
Intervention: Project executed cells as real execute_python tool calls paired with tool results; change nothing model-facing otherwise.
Prediction: Both arms pass G0; CodeAct queries on G1 call 1 and returns on call 2.
Measure: Exact answer/refs/query, valid return, transport class, calls/tokens/cells/errors/time.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; execution disabled.
Stop: First matched failure, integrity/route drift, cap, budget refusal, or mutation.
Invalid if: Projection/executor/controller commitment drift, unmatched route/facts/authority, missing trace, key disclosure.
Result map: G0 -> protocol; G1 -> transport/capability; G2+raw -> representation; G3 -> eligibility; pass -> consider shadow screen.
Outputs: analysis/parity-compat-v4/runs/<run-id>/
