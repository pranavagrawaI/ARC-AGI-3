# 0159-muse-multiprovider-compaction-ls20-v21
Question: Can persistent Muse compact before its observed limit and continue the same LS20 game with provider failover?
Posture: feasibility after v20 found six raw turns exceed the 131072-token endpoint limit
Contrast: none
Prediction: turns 0-4 act; one validated compaction retains raw turns 2-4; turn 5 returns one legal action.
Measure: summary, compaction transaction/checkpoint/archive, transcript, trajectory, provider events, recording.
Cap: 1 game; 6 actions; 14 model calls/turn; 90 s/provider attempt; 4 qualified providers; 60 min wall; unknown spend.
Stop: after action 6, terminal game, compaction failure, exhausted provider pool, or wall cap.
Invalid if: no compaction, wrong model/game, missing provenance, duplicate boundary action, or cap breach.
Result map: positive -> accept seam and design token trigger; negative -> diagnose stage; ambiguous -> tighten probe.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/multiprovider-compaction-v21/
