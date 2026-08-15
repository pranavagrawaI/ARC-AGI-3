# 0159 Muse human action labels LS20 v15
Question: Does the ARC human control vocabulary remove redundant action decoding?
Posture: public-interface correction; evidence preservation.
Contrast: v14 plus UP/DOWN/LEFT/RIGHT/SPACEBAR/CLICK x y/UNDO/RESET labels; no solver, evidence, memory, or policy change.
Prediction: Muse uses supplied controls directly and spends no calls inferring numeric mappings.
Measure: `summary.json`; transcript; provider events; exact decisions and trajectories.
Cap: 44 ARC actions; 14 provider calls/turn; one online LS20 game.
Stop: level complete, cap, failure, or operator abort.
Invalid if: wrong model/provider/source, scripted client, unenforced sandbox, uncertain ARC action, host repair/inference/fallback, numeric SDK IDs exposed to Muse, or resumed run.
Result map: labels work -> retain and add executable world model; transport fails -> repair labels only; ambiguous -> inspect preserved trace.
Outputs: run artifact directory.
