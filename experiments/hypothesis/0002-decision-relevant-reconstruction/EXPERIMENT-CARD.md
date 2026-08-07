# 0002-operational-ls20-step13-v1
Question: Can one Oracle Coach reconstruction traverse measurement, replay, search, certification, and held-out comparison deterministically?
Posture: tracer bullet
Contrast: identical rerun
Prediction: reruns emit byte-identical scene, replay, and plan artifacts; action 1 predicts exact public successor fields.
Measure: SHA-256 and JSON equality for those artifacts; comparison.json exact_matches.
Cap: 1 trace; 13 prefix observations; 1 held-out transition; 8 search expansions; 0 provider calls; 0 environment actions; 30 s.
Stop: pass once; abort on digest mismatch, held-out leakage, schema error, or cap breach.
Invalid if: held-out data enters pre-prediction machinery; source digest differs; result is reported as contestant evidence.
Result map: positive → inspect replay, then permit neutral-interface work; negative → repair that seam; ambiguous → preserve and discriminate.
Outputs: runs/<run-id>/
