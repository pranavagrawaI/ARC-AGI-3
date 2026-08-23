# 20260823T125133Z-deepseek-v4-flash-complete-screen
Question: Does CodeAct change retrieval/logic when arm exhaustion is scored, not fatal?
Posture: mutation-free shadow
Contrast: DeepSeek V4 Flash, Sail Research, reasoning on, same facts/schema/caps; native query/return vs Python query/return.
Prediction: Retrieval preserved; CodeAct improves >=2 failures; no success regression.
Measure: Retrieval, verdict/invariants, legality, failures, calls/tokens/queries/cells/time/cost.
Cap: 8 calls/session; 162 calls; 4096 output/call; 600 s/session; USD 2.978; .015 reserve/call; prior USD .02087208; total < USD 3.
Stop: Integrity/route drift, cap, mutation, or 24 sessions complete.
Invalid if: Drift; unmatched route/facts/schema/authority; missing trace; key disclosed.
Result map: Rule met -> larger shadow; retrieval loss -> fix/kill; equal cognition and <20% token reduction -> native; action disagreement -> live test; invalid -> no claim.
Outputs: ../../runs/20260823T125133Z-deepseek-v4-flash-complete-screen/
