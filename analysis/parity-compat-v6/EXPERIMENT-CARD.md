# parity-compat-v6
Question: Does a manifest-derived self-describing proxy let DeepSeek use metadata evidence?
Contrast: V6 vs preserved DeepSeek V5; only executable capability discoverability changes.
Prediction: Runtime inspection reveals the exact admitted query; CodeAct queries and returns within 3 calls.
Measure: Signature/doc/help, cells, raw/canonical queries, attempts, disclosures, return, transport, cost.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; execution disabled.
Stop: First matched failure, integrity/route drift, cap, budget refusal, provider failure, or mutation.
Invalid if: Contract/worker/provenance drift, unmatched route/facts/authority, missing trace, key disclosure.
Result map: G1 pass -> representation; G1 fail -> inspect discoverability trace; full pass -> shadow screen.
Outputs: analysis/parity-compat-v6/runs/<run-id>/
