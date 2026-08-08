# 0004 LS20 feedback planning E4B v10
Question: Can E4B complete the checkpoint by choosing one action per public successor?
Posture: offline closed-loop feasibility
Contrast: V7 was open-loop; v10 keeps its facts and adds feedback. V8 failed pre-call; V9 emitted malformed tool text. V10 explicitly selects the sole tool.
Prediction: 0/3 complete; feedback repairs blocks but does not reach trigger twice and target.
Measure: `trace.jsonl` actions/public outcomes; `episodes.json` engine completion.
Cap: 60 local calls; 129 offline actions including replay; 0 live; $0; 20 minutes.
Stop: episode on completion, terminal, or 20 actions; run at 3 episodes or first checkpoint/transport/parser error.
Invalid if: route/future/private leakage, checkpoint drift, missing response, or incomplete tool call.
Result map: 3/3 → remove affordance; 1–2 → replicate; 0 → inspect divergence, do not ablate; invalid → stop this tool interface.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
