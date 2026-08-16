# 0159-muse-multiprovider-compaction-ls20-v19
Question: Can persistent Muse cross forced compaction and continue the same LS20 game with provider failover?
Posture: feasibility, not gameplay benchmark or causal quality comparison
Contrast: none
Prediction: turns 0-7 act; one validated compaction precedes turn 8; turn 8 returns one legal action.
Measure: summary, compaction transaction/checkpoint/archive, transcript, trajectory, provider events, recording.
Cap: 1 game; 9 actions; 14 model calls/turn; 90 s/provider attempt; 4 qualified providers; 60 min wall; unknown spend.
Stop: after action 9, terminal game, compaction failure, exhausted provider pool, or wall cap.
Invalid if: no compaction, wrong model/game, missing provenance, duplicate boundary action, or cap breach.
Result map: positive -> accept seam and compare recovery; negative -> diagnose stage; ambiguous -> tighten probe.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/multiprovider-compaction-v19/
