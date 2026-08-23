# 20260823T124109Z-deepseek-v4-flash-latency-retry
Question: Does CodeAct change retrieval/logic versus native tools with a latency-safe session cap?
Posture: mutation-free shadow
Contrast: DeepSeek V4 Flash, Sail Research, reasoning on, same facts/schema/caps; native query/return vs Python query/return.
Prediction: Retrieval preserved; CodeAct improves >=2 failures; no success regression.
Measure: Retrieval, verdict/invariants, legality, errors, calls/tokens/queries/cells/time/cost.
Cap: 8 calls/session; 178 calls; 4096 output/call; 600 s/session; USD 2.986; .015 reserve/call; prior USD .01319596; total < USD 3.
Stop: Integrity/route drift, cap, mutation, or 24 sessions complete.
Invalid if: Drift; unmatched route/facts/schema/authority; missing trace; key disclosed.
Result map: Rule met -> larger shadow; retrieval loss -> fix/kill; equal cognition and <20% token reduction -> native; action disagreement -> live test; invalid -> no claim.
Outputs: ../../runs/20260823T124109Z-deepseek-v4-flash-latency-retry/
