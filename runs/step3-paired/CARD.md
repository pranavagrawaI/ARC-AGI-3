# Step 3 — paired wave: Duck alone vs Duck + critic (question mode), Kaggle, 1200 s

Frozen 2026-09-18 ~19:20 UTC, before either kernel was pushed. Tree `c81f3d3`
(critic package identical to `0fc7403`; the incumbent arm's prompts are byte-identical to
upstream 7652836 by snapshot test).

## Question
Does the question-mode critic change what Duck achieves per episode at the 1200 s cap?

## Arms (one kernel each, same tree, same bundle shape as Step 2 verify)
| arm | kernel | env |
|---|---|---|
| incumbent | `pranavagrawai/taaf-duck-step3-incumbent` | no `DUCK_CRITIC_*` |
| critic | `pranavagrawai/taaf-duck-step3-critic` | `DUCK_CRITIC_ENABLED=1`, `DUCK_CRITIC_MODE=question`, all else default |

## Design
| knob | value |
|---|---|
| games | sb26-7fbdac44, su15-1944f8ab, lf52-271a04aa, ft09-0d8bbf25, ls20-9607627b (Step 1's five) |
| passes | 4 → 20 episodes per arm, 2 waves at concurrency 10 |
| per-game cap | 1200 s |
| experiment cap | 75 min (2 waves × 20 min + slack); vLLM startup is outside it |
| accelerator | NvidiaRtxPro6000, internet off, Duck's own model + wheelhouse datasets |

## Primary metric and decision rule (precommitted)
Primary: **total levels completed** over the 20 episodes, per arm. Secondary: episodes with
≥1 level; seconds to L1 on sb26/su15; actions per episode; for the critic arm, firings,
critic wall %, uptake (probe executed next).

Submit rule: submit the arm with more total levels. Tie or within 1 level → submit the
**incumbent** (the critic is the unproven intervention; a wash does not earn the slot).
If either kernel errors before its games complete, it is re-pushed once; no result is
read until both arms have 20 episodes.

Prediction (written before data): incumbent ≈ 6–9 levels (sb26/su15 L1 saturated, sb26
sometimes L2, others rare); critic within ±2 of that. I expect a wash. A ≥3-level
difference in either direction would surprise me.

## Kill / invalidation
- Turn latency differs by >25 % between arms (~43–46 s expected) → substrate differed; rerun.
- Critic arm with 0 stall firings on ls20/lf52/ft09 → trigger did not run; invalid.
- Any game in the critic arm ended by a firing → invalid.
