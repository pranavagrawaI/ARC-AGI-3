# parity-compat-v5
Question: Can Kimi and DeepSeek use metadata evidence when meaning, attempts, and disclosures are separated?
Intervention: Canonicalize metadata queries without grid encoding; permit 2 attempts but only 1 successful disclosure.
Prediction: Both models repair or directly pass G1; no second evidence disclosure occurs.
Measure: Raw/canonical query, attempts, disclosures, return, transport, calls/tokens/errors/time.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; execution disabled.
Stop: First matched failure, integrity/route drift, cap, budget refusal, or mutation.
Invalid if: Normalizer/accounting/provenance drift, unmatched route/facts/authority, missing trace, key disclosure.
Result map: G0 -> protocol; G1 -> capability; G2+raw -> representation; G3 -> eligibility; pass -> shadow screen.
Outputs: analysis/parity-compat-v5/runs/<run-id>/
