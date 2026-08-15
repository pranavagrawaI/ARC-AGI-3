# 0159 Muse action-only commitment LS20 v5
Question: Can visible budget plus `submit_action` prevent Muse's per-turn tool exhaustion on LS20?
Contrast: v4 versus the same 0159 solver replacing raw structured JSON with one action-only tool.
Prediction: Muse reliably commits instead of returning empty final content.
Measure: levels, actions, calls, budget/commit events, usage, stop, source and trajectory hashes.
Cap: 44 ARC actions; one action per turn; 12 calls per turn: at most 11 analysis plus one `submit_action`; 8,192 output tokens per call.
Stop: level completion, game over/reset, cap, illegal/missing commitment, provider/worker/sandbox failure, or uncertain ARC action.
Invalidity: Wrong model/provider/source, analysis tool in commitment, host-selected action, mutable evidence, unenforced sandbox, resume, or missing successor.
Result to action: Preserve the run; fix technical/interface failures only. Add neither game semantics nor host fallback.
