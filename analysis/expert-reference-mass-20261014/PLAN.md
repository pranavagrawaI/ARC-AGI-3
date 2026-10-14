# Expert reference-mass screen

Authority: Pranav approved implementation and a short comparison on 2026-10-14, with one hour of total GPU use including startup. Primary outcome is more completed levels at equal total budgets. No automatic retry or promotion.

Plan: preserve c14 parent; add a hash-bound routing callback to target top5 only, normalize selected logits against top10, keep shared expert and top10 MTP unchanged. Verify probabilities against an independent float64 oracle and the exact packaged constructor seam. One private RTX PRO6000 session runs control then candidate from fresh game states for 15 minutes each, at c14 serving settings, within 3600 seconds including both startups. Same order/game roster and play time, separate files/caches/servers. Fixed order and one replicate are accepted limitations of the short screen.

Pre-mortem: a speed win could hide quality damage (compare levels/score, not acceptance); wrong callback could affect draft/shared output (target-only constructor guard and runtime layer markers); another runtime version could ignore the patch (full wheel/source hashes and installed readback); startup/teardown could consume the pair (absolute deadline, native hard timeout, require full equal play exposure or invalidate); residual requests could contaminate phase2 (stop/reap the owned process group and create a fresh benchmark/recording dir); logs could lose results on shutdown (incremental phase receipts and graceful reporting reserve).

Risk measurement: no CRAP gate/dependencies for this scope. Existing qwen2 constructor measured by AST complexity before/after; no branching added to constructor. New helper functions measured with fixed unittest statement-line trace; this is a local estimate, not production model coverage. GPU dispatch and whole-model behavior are separate runtime gates.

Paper: https://arxiv.org/html/2609.04575v1 . The chosen active5/reference10 mechanism is unproven for Flash-Next/ARC. This screen can reject a collapse or earn further testing; it cannot establish a replicated score gain.
