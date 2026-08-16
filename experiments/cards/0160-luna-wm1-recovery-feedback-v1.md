# 0160-luna-wm1-recovery-feedback-v1
Question: Will GPT-5.6 Luna use WM1 with visible broker results?
Posture: model-stack feasibility, not causal.
Contrast: Muse recovery-feedback run; same seed, interface, loop, and cap; model/provider differ.
Prediction: Luna reads WM1, authors `predict(grid, action)`, receives replay feedback, and later reads, revises, or cites it.
Measure: audit/events; cells, reads, edits, feedback, candidates, versions, trajectory, failures, levels.
Cap: 1 game; 8 actions; 14 calls/turn; 1 compaction/turn; 90 s/provider; OpenAI only; 60 min; unknown spend.
Stop: action 8, terminal, system failure, or wall cap.
Invalid if: wrong model/game/interface/provider, evidence/isolation fault, duplicate action, missing provenance, or cap breach.
Result map: adoption -> inspect validity/use; read-only -> redesign commitment; no use -> separate discoverability from utility; failure -> diagnose.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/wm1-recovery-feedback-luna-v1/
