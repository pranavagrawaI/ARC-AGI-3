# 20260823T123612Z-deepseek-v4-flash-turn-retry
Question: Does CodeAct change retrieval/logic versus native tools after recoverable malformed turns?
Posture: mutation-free shadow
Contrast: DeepSeek V4 Flash, Sail Research, reasoning on, same facts/schema/caps; native query/return vs Python query/return.
Prediction: Retrieval preserved; CodeAct improves >=2 failures; no success regression.
Measure: Retrieval, verdict/invariants, legality, errors, calls/tokens/queries/cells/time/cost.
Cap: 8 calls/session; 183 calls; 4096 output/call; 120 s/session; USD 2.991; USD .015 reserve/call; prior USD .008338575; total < USD 3.
Stop: Integrity/route drift, cap, mutation, or 24 sessions complete.
Invalid if: Drift; unmatched route/facts/schema/authority; missing trace; key disclosed.
Result map: Rule met -> larger shadow; retrieval loss -> fix/kill; equal cognition and <20% token reduction -> native; action disagreement -> live test; invalid -> no claim.
Outputs: ../../runs/20260823T123612Z-deepseek-v4-flash-turn-retry/
