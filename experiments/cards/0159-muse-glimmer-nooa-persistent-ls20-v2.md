# 0159 Muse Glimmer persistent NOOA LS20 v2
Question: Can one continuous Muse conversation complete LS20 within 44 actions when each turn receives the current image and legal actions?
Contrast: Replaces the one-turn plumbing probe; adds only persistence and execution, not semantic machinery.
Prediction: Muse preserves action results across turns well enough to complete at least one level.
Measure: levels_completed, actions, exact decisions and explanations, public frames, terminal reason, fixed worker identity.
Cap: 44 ARC actions; one action per turn; 8,192 output tokens per generation.
Stop: levels_completed >= 1, game over/full reset, action cap, illegal result, provider/worker failure, or uncertain ARC action.
Invalidity: Wrong model/provider, worker or Agent replacement, host-chosen action, missing successor, or resumed trajectory.
Result to action: Preserve the run. Fix technical/interface defects only; do not add semantic scaffolding from gameplay failure.
