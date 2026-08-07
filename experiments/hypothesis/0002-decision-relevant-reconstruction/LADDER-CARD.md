# 0002-gemma-relation-ladder-v1
Question: At what assistance level can Gemma select the LS20 L1 target/workspace pair?
Posture: competence ladder
Contrast: L5 verified relation values, then L4 explicit scale rule with uniform candidate measurements.
Prediction: L5 passes; L4 passes if Gemma can apply integer scale normalization.
Measure: each frozen request.json precedes response.json; parsed pair equals frozen answer 3.
Cap: 2 provider calls; 6000 output tokens/call; $1 total; 0 environment actions; 600 s.
Stop: abort on leakage/transport/malformed output; stop at first miss or after L4.
Invalid if: held-out answer enters either request, candidate order differs, or prompts change after first call.
Result map: L5 miss → selection failure; L5 pass/L4 miss → rule-application failure; both pass → test closed rule inventory next.
Outputs: runs/ladder-ls20-l1-<timestamp>/
