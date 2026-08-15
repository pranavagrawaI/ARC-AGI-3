# 0159 Muse CodeAct budgeted commitment LS20 v4
Question: Can visible budget plus reserved commitment prevent Muse's per-turn tool exhaustion on LS20?
Contrast: v3 versus the same 0159 solver with budget state exposed and commitment physically separated.
Prediction: Muse produces a legal model-selected action instead of exhausting the turn.
Measure: levels, actions, calls, budget/commit events, usage, stop, source and trajectory hashes.
Cap: 44 ARC actions; one action per turn; 12 calls per turn: at most 11 analysis plus one tool-free commitment; 8,192 output tokens per call.
Stop: level completion, game over/reset, cap, illegal/missing commitment, provider/worker/sandbox failure, or uncertain ARC action.
Invalidity: Wrong model/provider/source, analysis tool in commitment, host-selected action, mutable evidence, unenforced sandbox, resume, or missing successor.
Result to action: Preserve the run; fix technical/interface failures only. Add neither game semantics nor host fallback.
