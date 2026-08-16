# 0160-muse-wm1-recovery-feedback-v1
Question: With broker results automatically visible, will Muse author and use a quick executable theory?
Posture: feasibility; no behavior-change claim.
Contrast: none.
Prediction: a bare read exposes the quick template; after observing a transition Muse authors `predict(grid, action)`, receives replay feedback, and later reads, revises, or cites it.
Measure: audit/events; cells, reads, edits, feedback, candidates, versions, trajectory, failures, levels.
Cap: 1 game; 8 actions; 14 calls/turn; 1 compaction/turn; 90 s/provider; 4 providers; 60 min; unknown spend.
Stop: action 8, terminal, system failure, or wall cap.
Invalid if: wrong model/game/interface/provider, evidence/isolation failure, duplicate action, missing provenance, or cap breach.
Result map: adoption -> inspect validity/use; read-only -> redesign commitment; no use -> redesign visibility; failure -> diagnose.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/wm1-recovery-feedback-muse-v1/
