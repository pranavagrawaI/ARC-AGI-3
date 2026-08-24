# parity-compat-v9
Question: Does a flat retrieval-only terminal schema let Gemma 4 31B clear termination?
Contrast: V9 vs preserved Gemma V8; only terminal schema/prompt changes from wrapped four-type union to direct conclusion + evidence_refs.
Prediction: Both arms return READY correctly within three calls.
Measure: Typed returns, invalid interactions, calls, route, tokens, cost, integrity.
Cap: <=5 gates; 2 arms; 3 calls/arm/gate; <=30 calls; execution disabled.
Stop: First matched failure, integrity/route drift, cap, budget refusal, provider failure, or mutation.
Invalid if: Contract/controller/worker/provenance drift, unmatched route/facts/authority, missing trace, key disclosure.
Result map: G0 fail -> reject schema hypothesis; G0 pass -> metadata; representation outcomes remain separate.
Outputs: analysis/parity-compat-v9/runs/<run-id>/
