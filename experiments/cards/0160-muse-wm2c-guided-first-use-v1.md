# 0160-muse-wm2c-guided-first-use-v1
Question: Will exact guided first use make Muse inspect the lane and then edit or explicitly defer?
Posture: feasibility; no behavior-change or gameplay claim.
Contrast: none; cumulative WM2 plus condition-specific read instruction.
Prediction: repeated-action evidence triggers `self.read_world_model()`; Muse then makes a narrow edit, records prose uncertainty, or explicitly defers while returning legal actions.
Measure: exposure, reads, edits, evaluations, deferrals, Python calls, failures, latency, trajectory, levels.
Cap: 1 game; 8 actions; 14 calls/turn; 1 compaction/turn; 90 s/provider; 4 providers; 60 min; unknown spend.
Stop: action 8, terminal, system failure, or wall cap.
Invalid if: wrong treatment/model/game/provider, missing commitments, containment failure, duplicate action, or cap breach.
Result map: read+edit -> inspect artifact; read+defer -> inspect judgment; read+scratchpad -> editing friction; no read -> task-delivery failure; failure -> diagnose.
