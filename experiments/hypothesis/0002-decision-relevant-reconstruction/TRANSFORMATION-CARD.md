# 0002-gemma-transformation-microbattery-v1
Question: Can Gemma identify executable transformation relations from complete symbolic examples?
Posture: competence microbattery
Contrast: four frozen relation cases: scale-vs-tile, congruence, colour/embed, modular progression.
Prediction: all four cases return the expected relation and parameters.
Measure: parsed final JSON equals each case's frozen expected object; raw request/response retained.
Cap: 4 provider calls; 2000 output tokens/call; $1 total; 0 environment actions; 600 s.
Stop: one call per case; abort on transport failure or cap breach.
Invalid if: expected answer enters a request or cases change after the first call.
Result map: 4/4 → test LS20 closed-inventory selection; any miss → isolate that relation/interface.
Outputs: runs/transformation-microbattery-<timestamp>/
