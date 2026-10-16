# Model change, 2026-10-16: Tinfield is not a better reasoner per token; public checkpoints are exhausted

No GPU. `matched.py` re-reads the Tinfield r3 smoke (kernel 138493302 v1, 2026-10-13) and both c14 controls with
r3's own clock definitions, then compares levels at matched cumulative completion tokens (`matched.json`).

## The r3 gate was confounded; the verdict survives

All 25 Tinfield games ran the full ~40 min (`final_wallclock_seconds` 2400–2520), but the transcript clock starts
at each game's first finished turn, up to 20 min late under queueing (KV 0.91, queue 2). Late-starting games
are observed only to game-minute ~20. So "38 levels at minute 30 vs 55" compares mostly-truncated games against
complete ones. Valid window: game-minute ≤ 20, ≈ 1.0M tokens.

| Levels at matched tokens | Tinfield NVFP4 | c14 wave-1 | c14 pruning control |
|---|---:|---:|---:|
| 0.50M | 15.6 | 17.1 | 19.5 |
| 0.75M | 21.5 | 27.5 | 33.9 |
| 1.00M | 31.0 | 38.1 | 41.4 |

Tinfield is below both controls at every point, about 20% behind at 0.75–1.0M. The two controls differ by about
10%. Tinfield is not ahead per token, and it also generates fewer tokens on our stack (608.6 vs 818.8 tok/s).
Confound: NVFP4 + INT8 PLE vs the controls' W4A16. A format control (base Flash-Next NVFP4, e.g.
`michaelpoluektov/qwen3-8-flash-next-nvfp4`) would only split the deficit between format and model. No outcome
of that run produces a winner, and there is no W4A16 Tinfield build (we cannot quantise 177B locally). Not run.

## Field check, 2026-10-16

Web and Kaggle searches: no Qwen4 weights, no Flash-Next post-train since the 10-13 survey, and nothing new on
Kaggle with an ARC signal. The remaining shortlist entries are a contaminated flat-loss LoRA (huikang v15,
NVFP4), AZ-Focus (graft blocked; its INT4 sibling sits on the top-5 base that lost 14 vs 44 levels), and Swift
(lost 47 vs 101.5). None has evidence of a per-token judgment gain.

Conclusion: the model-change lever is closed among public checkpoints. A judgment gain would have to come from
training, which the 2026-10-10 decision excluded on cost.
