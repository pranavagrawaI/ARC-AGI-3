# 0159 Muse CodeAct trajectory LS20 v3
Question: Does read-only computation over executed transitions help persistent Muse complete LS20 within 44 actions?
Contrast: `predict` baseline versus `codeact`; host, model, provider, images, and action conditions remain matched.
Prediction: CodeAct tests action effects or geometry and completes at least one level more reliably.
Measure: levels, actions, decisions, usage, source/trajectory hashes, CodeAct cells, worker identity, validity/stop.
Cap: 44 ARC actions; one action per turn; 12 CodeAct iterations; 8,192 output tokens per generation.
Stop: level completion, game over/full reset, cap, illegal result, provider/worker failure, sandbox failure, or uncertain ARC action.
Invalidity: Wrong variant/model/provider, host-chosen action, mutable host evidence, unenforced sandbox, missing successor, resume, or source change.
Result to action: Preserve both runs; fix technical/interface defects only before considering a world model.
