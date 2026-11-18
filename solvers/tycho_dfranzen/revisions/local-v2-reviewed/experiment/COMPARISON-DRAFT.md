# Whole-system comparison — draft, not a run card

Gated on gpu-q4 runtime qualification and inspection of actual provider/tool/action evidence. No launch is authorized by this draft alone; campaign authority is already recorded in PLAN.md.

Use fresh private notebook runs of the donor and hybrid, with identical serving cells, model/drafter versions, GPU type/count, game versions/list/order, 25 concurrent host games and donor gate of 10. Both use the same 128 Ki input/output envelope and sampling parameters already inherited from the donor. Solver internals differ by design.

Hard cap: 7200 allocated seconds per arm; soft end 6600 seconds after notebook start to retain artifacts before hard cutoff. Per-game cap 5400 seconds, including admission wait. One pass over all 25 public games, no action-count cap. Reserve two GPU-hours per arm, including startup. Start fresh servers, record startup and ready timestamps, and compare score at both equal allocated wall time and common post-readiness exposure. If unequal startup prevents the latter, retain it as a system-level cost and do not call the result a pure reasoning effect.

Before freeze, inspect exact donor timeout semantics and scheduler clock against the hybrid; nominally identical config values are insufficient. Verify whether per-game clocks begin before queue admission and whether host concurrency starts all games at once. If semantics differ, use an explicit common absolute game deadline in both arms rather than hiding unequal exposure behind the same number.

Choose order using a recorded random draw before launch. One pair is a screening result, not a robust estimate across model sampling variation. Never compare this arm with the historical donor score: that history has different exposure and cannot serve as a matched control.

Primary result: paired official score and levels; retain failures, incomplete games, action counts, wall time, per-role tokens and unknown usage. Diagnostic mechanisms: prospectively supported first hypothesis and time/actions to repair after contradiction. Missing prospective evidence is unmeasured, not correct.

After the pair, choose at most one ablation based on a failure or useful mechanism visible in raw traces. A whole-system win without the ablation leaves mechanism attribution unfinished. A score loss with better mechanism metrics is a promising direction only. Freeze a new contrast card before observing its results; preserve all prior sources/results.
