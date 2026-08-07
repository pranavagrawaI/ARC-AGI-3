# 0002-neutral-acquisition-ls20-l1-v1
Question: Can Gemma select the held-out-confirmed relation from a prefix-only neutral digest without frames, game identity, actions, outcome, or Oracle labels?
Posture: acquisition feasibility gate
Contrast: Gemma selection versus frozen random, biggest, busiest, and scale rules
Prediction: Gemma selects the unique retrospectively confirmed pair on intact LS20 L1.
Measure: request.json excludes forbidden fields; response.json pair equals score.json answer.
Cap: 1 cell; 1 provider call; 6000 output tokens; $1; 0 environment actions; 300 s.
Stop: stop after the first scored response; abort on leakage, transport failure, malformed output, or cap breach.
Invalid if: outcome or Oracle answer enters the request; candidate policy changes after response; source trace is not the frozen prefix/full split.
Result map: hit → build perturbation probe; miss → diagnose selector input; invalid → repair authority boundary.
Outputs: runs/acquisition-ls20-l1-<timestamp>/
