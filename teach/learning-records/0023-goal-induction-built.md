# Goal induction is clause search relabelled to the reward channel

Lesson 0015 (`teach/lessons/0015-goal-induction.{py,html}`) builds goal induction
as the same 0012 searcher with the consequent relabelled from "actor body
changed" to "the success channel advanced" (`levels_completed` rose). `learn_goal`
adds two honest refusals so it never fabricates a win condition: `GRAMMAR_GAP`
(rewards differ but no clause separates them) and `NO_GOAL_EVIDENCE` (no rewarded
step seen, or every step looks rewarded). 5/5 offline checks pass.

The pedagogical spine is a deliberate corpse (LR_0016): ls20's goal is a *match*
(actor attribute equals a target's, then stand on it), which absolute symbols
(`target_shape_*`, `actor_shape_*`) cannot express — a conjunction can test
symbol presence, never symbol *equality*. So check 3 returns `GRAMMAR_GAP` even on
perfect reward data. That proven gap earns exactly one new letter: a generic
relational `match_<slot>` built from 0010's `equal`/`cheapest_equal` (lens chosen
by the invariance lattice: raw before rotation; recolour = `color_invariant`).
With it, check 4 recovers `IF match_<slot> AND on_target THEN reward`.

LIVE (400 steps, seed 17) is a **double fail-loud**, and that is the finding:
- controllability re-derived `(9,15)+(12,10)`; 114 goal-candidate events on 14
  shapes logged.
- **reward steps = 0** → `learn_goal` returns `NO_GOAL_EVIDENCE` (no planner →
  never satisfies the win condition by luck). The refusal-instead-of-hallucination
  is the positive epistemic result.
- **match letter fired = 0 (verified in both composite and parts mode)** → the
  avatar is a 2-colour composite and tiles are single-colour, so the avatar's raw
  body never equals a tile under any lens. ls20 matches a *carried attribute*, not
  the avatar's pixels.

Two corpses, two named next bricks: (1) reward-seeking planning over the induced
dynamics so `levels_completed` rises on purpose (the only way to feed positives to
`learn_goal`); (2) a slot-gap not on the lens but on the **comparand** — the
alphabet needs a learned answer to "which objects get compared?", deferred to
lesson 0016. Honest boundary: no level was cleared; the goal-induction *machine*
is verified, ls20 starves it without directed exploration.
