# 0160-muse-wm1-quick-theory-v1
Question: Can Muse adopt a low-threshold partial predictor and use its replay feedback without breaking the actor loop?
Posture: feasibility; no behavior-change claim.
Contrast: none.
Prediction: after observing a transition, Muse authors `predict(grid, action)`; a valid candidate gets bounded replay feedback and is later read, revised, or cited in a decision.
Measure: audit/events; candidates, evaluations, versions, trajectory, failures, levels.
Cap: 1 game; 8 actions; 14 calls/turn; 1 compaction/turn; 90 s/provider; 4 providers; 60 min; unknown spend.
Stop: action 8, terminal, system failure, or wall cap.
Invalid if: wrong model/game/interface/provider, evidence/isolation failure, duplicate action, missing provenance, or cap breach.
Result map: adoption -> inspect validity/use; no adoption -> redesign visibility; failure -> diagnose before rerun.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/wm1-quick-theory-muse-v1/
