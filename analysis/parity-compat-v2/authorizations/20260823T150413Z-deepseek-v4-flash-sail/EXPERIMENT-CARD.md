# parity-compat-v2-deepseek-sail
Question: Can DeepSeek V4 Flash complete both Native and sterile CodeAct eligibility protocols?
Contrast: Same model/checkpoint/schema/caps; native query/return vs sterile Python query/return.
Prediction: Both pass termination, metadata, representation, and easy retrieval within 3 calls/arm/gate.
Measure: Exact answer/refs/query, typed return, calls/tokens/cells/errors/time, durable route and cost.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; 1,048,576 input and 4,096 output/call; <=USD 3.
Stop: First matched gate failure; integrity/route/provider/call/cost failure; any mutation; ladder complete.
Invalid if: Commitment or route drift; missing trace/usage; key disclosure; extra authority.
Result map: G0 -> reject protocol; G1 -> capability; G2+raw -> representation; G3 -> cognitive eligibility; pass -> eligible only.
Route: OpenRouter Sail Research sail-research/fp4; reasoning enabled; no fallbacks.
