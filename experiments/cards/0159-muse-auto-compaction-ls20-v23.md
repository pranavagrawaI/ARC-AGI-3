# 0159-muse-auto-compaction-ls20-v23
Question: With the endpoint 131072 window configured, does 80% auto-compaction continue LS20 near 30%?
Posture: repair after v22 exposed an unknown-window integration defect
Contrast: none
Prediction: after a completed turn reaches 80% of usable context, host compacts before the next turn, keeps a complete-turn suffix, and the next action succeeds at <=30%.
Measure: decision.context, transcript compaction row, transaction, checkpoint/archive, trajectory, provider events, recording.
Cap: 1 game; 8 actions; 14 model calls/turn; 90 s/provider; 4 providers; 60 min; unknown spend.
Stop: action 8, terminal, compaction/provider failure, or wall cap.
Invalid if: no trigger, mid-turn boundary, wrong model/window/game, missing provenance, duplicate action, or cap breach.
Result map: positive -> retain policy; >30% -> revise suffix estimate; failed continuation -> diagnose; ambiguous -> tighten.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/auto-compaction-v23/
