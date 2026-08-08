# 0004 LS20 L2 closed-reference acquisition E4B v3

Question: Does deduplicated prefix evidence suffice for E4B with closed evidence references?
Posture: offline acquisition gate
Contrast: V1 had 7 hypotheses and abstained; v2's open reference field failed parsing.
Prediction: 3 calls select pair 32, scale factor 2, with one stable signature.
Measure: `raw-responses.json` calls; `responses.json` contract, prefix, and hidden support.
Cap: 3 local E4B calls; 0 environment actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/model-drift error.
Invalid if: scorer leakage, drift, missing raw response, or incomplete response.
Result map: stable supported selection → offline planning gate; abstention → evidence insufficient; wrong/unstable → diagnose; invalid → repair/new card.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
