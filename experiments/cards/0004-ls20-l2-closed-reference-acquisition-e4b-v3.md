# 0004 LS20 L2 closed-reference acquisition E4B v3

Question: Does deduplicated prefix evidence suffice for E4B to adopt the acquired objective when evidence references use a closed record?
Posture: offline acquisition gate
Contrast: V1 abstained with seven hypotheses; v2 was invalid because an open evidence-reference field failed parsing.
Prediction: All 3 calls select pair 32, `integer_scale_objective`, factor 2, with one stable signature.
Measure: `raw-responses.json` tool calls and `responses.json` contract, prefix semantics, and hidden terminal support.
Cap: 3 local E4B calls; 0 environment actions; $0; 10 minutes.
Stop: After 3 calls or first transport, parser, or model-drift error.
Invalid if: scorer leakage, source/model drift, missing raw response, or incomplete response.
Result map: stable supported selection → proceed to offline planning gate; abstention → evidence remains insufficient; wrong/unstable → diagnose acquisition; invalid → repair under a new card.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/0004-ls20-l2-closed-reference-acquisition-e4b-v3-<UTC>/`
