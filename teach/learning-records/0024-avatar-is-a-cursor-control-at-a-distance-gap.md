# The avatar is a cursor, not the state — and the grammar can't say so

Instrumenting live ls20 (interaction only, source unread) killed a hypothesis we
both held: we set out to fix avatar-attribute occlusion and found the avatar has
nothing to occlude.

## Measured facts
- Avatar body `(9,15)+(12,10)` is read 100% of 600 steps and **never changes
  appearance** (1 palette, 1 mask, 0 changes). The controllable body has no
  variable shape/colour. Therefore `match_fire=0` was never an occlusion bug — we
  were comparing the avatar's body to floor tiles, two things that can't be equal
  and neither of which is the matchable object.
- The matchable state lives in **HUD/indicator sprites**: a top box (blue shape,
  area 5, at the end of a corridor — the target/door template) and a bottom-left
  box (blue shape, area 20 — the workspace/current display). Both are colour-9.
- **Puzzle is STABLE across 8 resets** (both boxes identical every reset). Reset is
  a free budget refill of the *same* puzzle → a sweep may span episodes.
- **Hard ~129-action budget**: GAME_OVER fires around step 129 (move-bar/lives).
  A 24-cell × ~10-action blind sweep (~240 actions) exceeds one episode; stable
  resets make a multi-episode sweep viable, RHAE≈0 (fine for a first win).
- The top box (5px) and bottom-left box (20px) differ ~4× in scale. If those are
  the two comparands, matching needs **scale-invariance — a lens 0010 does NOT
  have** (it has translation/rotation/reflection/colour only). Possible third gap;
  unconfirmed (the true comparand pair is not yet established).

## The instrument was lying (fixed)
`run_goal_probe` reset+`continue`d on WIN/GAME_OVER *before* reading
`levels_completed`. Since the level counter ticks on the WIN frame itself, the code
could win and record nothing. Fixed: read `reward_up` on every frame (incl.
terminal) before the reset, log the event there, and wire GAME_OVER as a negative
bit. Verified: `death_steps` now > 0 (was structurally 0). Death validates
**plumbing only** — it is proximal (bar hit zero); a win is distal (construction
30 steps earlier), so a clean death-catcher does NOT prove the win-catcher works.

## The real GRAMMAR_GAP (representational, not perception) — the next 0016
ls20's win is "object A's attribute equals object B's, where A is mutable by my
actions and B is not." Auditing the alphabet:
- **Cross-object equality `A.attr == B.attr`: EXPRESSIBLE.** 0010 `equal(slot,a,b,
  under)` already compares any two ObjectViews; `match_facts` uses it. It is only
  *wired* avatar-vs-target; generalising to sprite-vs-sprite is trivial.
- **Control-at-a-distance "action by X changes attribute of Y": NOT expressible.**
  0008 learns only `samples[sig][action] -> [(dr,dc)]` — action → *translation of
  the moved object itself*. 0009's `changed?` is avatar-egocentric (LR 0015 already
  flagged non-local switch→door effects as silently absorbed). The cause-on-X →
  effect-on-remote-Y link has no slot. The control mechanism for ls20 is unsayable.

So the stack was implicitly typed for "navigate avatar to goal tile"; ls20 is
"avatar is an instrument that mutates a remote sprite until it matches a template."
0016 = extend the alphabet to (a) remote-CHANGER attribution (avatar action →
attribute change of a non-actor sprite) and (b) two-non-avatar-sprite relational
equality tied to reward. Then the "sweep" becomes "read template B, drive workspace
A to match it" — guided, fits any budget. Discipline: do NOT hardcode `A==B` for
ls20; add general cross-object equality + remote-CHANGER detection and let them
fire on the sprites the agent discovers (mutable workspace vs invariant template).
