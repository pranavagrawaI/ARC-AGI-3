# parity-compat-v12

Question: do identifier binding and RLE interpretation work separately, and fail only when composed?

Contrast: matched Native and sterile CodeAct; same immutable component inputs, route, three-call ceiling, and flat returns; only G1/G3 can query evidence.

Prediction: G2A/G2B pass independently; G3 failure after both pass identifies composition interference.

Measure: typed value, identifier-error class, Python cells, provider calls, return-call index, tokens, invalid interactions, wall time.

Cap: 5 gates, 3 calls/arm/gate, 30 total; execution disabled until separately authorized.

Stop: complete a started pair, then first failure; immediately on integrity, route, provider, or budget failure.

Invalid: input drift, key disclosure, unmatched route, missing trace/accounting, extra authority, canonical mutation.

Actions: G2A fail -> binding; G2B fail -> representation; G3 fail -> composition; all pass -> parity corpus eligible.
