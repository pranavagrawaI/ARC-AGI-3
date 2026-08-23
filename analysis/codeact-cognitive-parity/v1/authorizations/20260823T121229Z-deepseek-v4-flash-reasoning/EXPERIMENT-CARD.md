# 20260823T121229Z-deepseek-v4-flash-reasoning
Question: Does CodeAct change retrieval/reasoning versus native tools?
Posture: mutation-free shadow
Contrast: Same DeepSeek V4 Flash, Sail Research, reasoning on, facts/schema/caps; native query/return vs sterile Python query/return.
Prediction: Retrieval preserved; CodeAct improves >=2 failures; no success regression.
Measure: Per-arm result/provenance: retrieval, verdict/invariants, legality, errors, calls/tokens/queries/cells/time/cost.
Cap: 24 sessions; 8 calls/session; 192 calls; 4096 output tokens/call; 120 s/session; USD 3 total; USD .015 reserve/call.
Stop: Integrity violation, route drift, cap, mutation, or 24 sessions complete.
Invalid if: Drift; unmatched route/facts/schema/authority; missing trace; key disclosed.
Result map: Rule met -> larger shadow; retrieval loss -> fix/kill; equal cognition and <20% token reduction -> native; action disagreement -> live test; invalid -> no claim.
Outputs: ../../runs/20260823T121229Z-deepseek-v4-flash-reasoning/
