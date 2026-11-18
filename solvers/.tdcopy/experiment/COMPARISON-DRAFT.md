# Small-set whole-system comparison — draft, not a run card

**2026-10-04: game set, margin, result map, charging and launch are decided; see "Pair decisions" in PLAN.md. Where this draft disagrees, PLAN.md wins.**

Same FlashNext model, a fixed three-game set, one pass, three concurrent games, donor gate of 10, no action-count cap. The previous public-25 proposal is superseded. Each arm is a fresh private run with a frozen card and manifest. No public publication or competition submission.

Provisional comparison set: ls20/re86/sc25, subject to inspecting gpu-q5 before freeze. Keep ar25/ls20/sb26 for qualification. Historical control and control-r2 finish ar25 at score 100 and sb26 at 94.43/97.91; the S1 prompt treatment finishes both at 100. Thus completing all levels does not necessarily saturate score, but these two offer little observed final-score upside. At 3600 seconds on the actual benchmark game clock, re86 scores 41.67/41.67 in the two controls and sc25 scores 47.62/47.62. ls20 retains the hard case at 3.57/3.57. This supports a selection with more headroom; it does not establish equivalent difficulty under three-game scheduling. See score_headroom.py and evidence/score-headroom.json for hashed sources and reproduced scores. S1 is a treatment, not a third identical control.

The earlier levels_at_time.py starts its clock at the first transcript turn; its “60 minutes” cannot be described as 60 minutes including admission/readiness. The replacement analysis joins event action_num to benchmark history wallclock_seconds. Neither historical clock is notebook allocated GPU time. The planned pair must retain that separate allocation clock.

Provisional cap: 5400 allocated seconds per arm (1.5 GPU-hours on one RTX PRO 6000); soft stop 5100 seconds after notebook start; 3600 seconds/game including readiness/admission. Qualification has separate smaller caps. Record server-ready, game-start and action timestamps. Primary score comparison uses a common allocated-time cutoff; post-readiness curves are diagnostic, not a replacement for startup cost. Tokens/calls need not be equal: they are costs/outcomes under the same time constraint.

Before freeze, verify both deadline/cancellation implementations. Both source paths start game clocks before admission, but donor uses its own thread pool and hybrid uses asyncio's default executor. Three games with ten slots avoid queued scheduling; this does not qualify public-25 fairness. Audit donor warmup RESET and retain its cost as part of the whole-system bundle rather than silently normalizing it away afterward.

Use identical immutable weights/drafter, serving cells, GPU/container and sampling. Record a random draw for arm order before launch. Budget the entire pair first. Retain failures, incomplete games, interrupted requests and unknown usage. Report per-game score/levels and score versus allocated time, not only the mean. This is screening on three inspected games, not an estimate for the full public/hidden distribution. One pair remains provisional under sampling variation.

Inspect the actual chain: tool offered -> Actor request -> Builder admitted -> model changed -> executable checks -> prospective prediction -> engine observation -> revision -> verified corrected prediction. Report denominators throughout. Successful tool delivery, Python syntax repair or model ABI acceptance is not correct game-mechanism inference.

Choose one intervention from discovery traces, then freeze its hypothesis, changes and interpretation before a fresh pair. Do not switch TYCHO_MODE and call it a single-tool ablation: it also changes ownership, seeding, feedback and prompts. Builder-only removal tests delegation availability; native Python can still support dynamics computations. Scope the claim accordingly.

If construction is unused, record non-use and skip an inert ablation. Better mechanism diagnostics with lower score identify a promising direction, not a better solver. Sparse/stochastic evidence can remain inconclusive. The mechanism question is unfinished unless the actual contrast supports it.
