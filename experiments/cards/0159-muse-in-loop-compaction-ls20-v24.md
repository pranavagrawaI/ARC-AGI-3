# 0159-muse-in-loop-compaction-ls20-v24
Question: Can a context rejection compact and retry the same LS20 turn safely?
Posture: bounded failure-path feasibility
Contrast: none; trigger set to 99.9999% to exercise rejection recovery
Prediction: provider rejects an oversized request; worker returns needs_compaction without deleting history; host compacts; the same turn returns one action; ARC receives it once.
Measure: transcript and provider events, transaction, trajectory, recording.
Cap: 1 game; 8 actions; 14 calls/turn; 1 in-loop compaction/turn; 90 s/provider; 4 providers; 60 min; unknown spend.
Stop: action 8, terminal, second overflow, compaction/provider failure, or wall cap.
Invalid if: no rejection, implicit archival, duplicate/missing action, wrong model/game, missing provenance, or cap breach.
Result map: positive -> retain recovery; negative -> diagnose; ambiguous -> tighten.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/in-loop-compaction-v24/
