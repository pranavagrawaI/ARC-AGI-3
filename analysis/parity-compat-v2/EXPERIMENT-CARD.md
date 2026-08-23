# parity-compat-v2
Question: Can one model reliably terminate, query, consume exact grids, and solve a trivial retrieval task in both Native and sterile CodeAct?
Contrast: Same model/facts/schema/ceilings; native query/return vs sterile Python query/return.
Prediction: Both pass each prerequisite within 3 calls; RLE preferred, raw only after RLE failure.
Measure: Exact answer/refs, required query, valid typed return, calls/tokens/queries/cells/errors/time.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; spend unset; execution disabled.
Stop: First matched gate failure, integrity/route drift, cap, budget refusal, or any mutation.
Invalid if: Commitment drift, unmatched route/facts/schema/authority, missing trace, key disclosure.
Result map: G0 fail -> reject protocol; G1 -> capability; G2+raw -> representation; G3 -> cognitive eligibility; pass -> separately authorize 12-packet shadow.
Outputs: analysis/parity-compat-v2/runs/<run-id>/
