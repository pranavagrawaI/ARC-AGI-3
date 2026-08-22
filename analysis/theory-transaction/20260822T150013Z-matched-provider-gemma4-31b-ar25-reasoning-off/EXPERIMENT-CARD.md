# Transactional Builder provider replay 2
Question: At AR25 10/18 typed exact (legacy 8/16), does the transaction prevent obligation loss on the intended live route?
Posture: corrected matched zero-ARC mechanism gate.
Contrast: 12-step control vs transaction; same frontier, Gemma 4 31B, CoreWeave, task, 4096 tokens; reasoning disabled as in the live treatment.
Prediction: transaction assesses an edit and persists no exact-transition loss.
Measure: typed exact deltas, model bytes, decision, ledger, notes/report, route/cost.
Cap: 14 calls/arm, 28 total, 290 s/arm, $0.42; reserve $0.015/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider error, or route drift.
Invalid if: commitments drift, no treatment edit is assessed, route/cost unavailable, or boundary exceeded.
Result map: valid no-regression -> one 30-action AR25 treatment; regression/deadlock -> stop; invalid -> stop and revise machinery.
Outputs: this directory.
