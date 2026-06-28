# Win falsifier: NOT VERIFIED — scale-norm match is a false positive at spawn

Ran the win falsifier (cycle A at cross → visit door cells → read `levels_completed`
with line-378 ordering). **No level tick observed** across ~1040+ steps and multiple
strategies. The instrument did not discard terminal frames; death was caught in
earlier runs. This falsifies the *sufficient* hypothesis "coarse scale-match + door
approach ⇒ win."

## What was tried (localizable)
1. **v2:** spawn already `scale_match=True` (12×12 norm) → BFS to top corridor/door
   floor cells. Reached door zone 76 step-visits. `reward_ticks=0`.
2. **v3:** 12 rounds of cross-hammer + door visits at `(17,36)…(14,36)`. Only 3
   distinct A states (cross hammer weak). 15 door visits. `rewards=0`.

## The important negative (confirms Gap B severity)
At spawn, comparing workspace A (bottom-left box) vs template B (top box):
- **Raw equality (any rotation): False**
- **Scale-normalized equality (16×16 resample): True**

So the coarse scale lens declares "matched" when the game clearly does not agree
(no win on door entry). This is exactly the trap: a predicate that's structurally
callable but semantically blind. Win needs a **stricter** scale-aware match than
8×8/12×12/16×16 nearest-neighbour resampling — or additional attributes (exact
rotation, colour channel, standing on a specific door tile inside the walled chamber).

## What is still unknown (next falsifiers)
- Whether ANY A state under a **stricter** scale lens + correct rotation wins on
  door entry (brute all ~6 A rotations at cross, then door).
- Whether additional changer tiles exist (orange/blue blocks below top box) for a
  second axis (colour vs shape) — cross alone gave only 3 A states in v3.
- Whether the win tile is not reachable floor at `(17,36)` but requires entering
  the walled door cell (may need different navigation target).

## Verdict
**NOT VERIFIED.** Pipe is trustworthy (line-378 fixed; deaths caught). Reachability
to corridor is achievable. Win is not "scale-match + stand near door" under current
match predicate. Gap B is not just "need a scale lens" — need one **tight enough**
to be False at spawn when raw match is False.
