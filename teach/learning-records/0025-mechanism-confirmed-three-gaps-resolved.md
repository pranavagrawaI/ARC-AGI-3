# One dominant probe resolved all three ls20 gaps — confirm-before-design wins

Pranav: confirm the mechanism before designing the grammar (orders-of-magnitude
cheaper; and you can't shape the slot without knowing the trigger type). The
probe is strictly dominant. One reset-aware, reachability-logged, full-coverage
run answered Gap-A-exists, A's trigger-type, and Gap-B-necessity at once.

## The reachability false-negative (the lesson inside the lesson)
First probe used least-visited-neighbour movement: 34 cells, stepped on the white
cross ONCE, reported "Gap A = 0 remote changes." That negative was untrustworthy —
the trigger was barely visited. Switching to lesson 0006 `SystematicExplorer`
(BFS-to-frontier) gave FULL room coverage (36 cells ≈ the whole level-1 room) and
stepped on the cross 10×. Only then is a negative trustworthy. A "no effect" result
is only as good as the proof its triggers were visited.

## Gap A — control-at-a-distance: CONFIRMED REAL
With full coverage, the avatar body `(9,15)+(12,10)` changed appearance **0 times**;
**3 steps changed a NON-avatar sprite**, always the same region (rows 55–56, cols
5–6 = the bottom-left HUD box), and always when the player was on/at the white cross
`~(31–36, 21–26)`. Before/after frames: stepping on the cross **rotated the
bottom-left box's blue L-shape**; the top box stayed fixed. So:
- **Workspace A = bottom-left HUD box** (mutable, controlled by the cross).
- **Target B = top box** in the door chamber (invariant; stable across resets).
- The cross is a CHANGER whose effect lands on a REMOTE object, not the avatar —
  exactly the non-local effect 0009 silently absorbs (LR 0015).

## A's trigger type — POSITIONAL/CONTACT, not a button
ls20 actions are `[1,2,3,4]` only (no ACTION5). The cause is **contact with a
specific tile-class (the cross)**, not a key. So the grammar slot is NOT 0008's
`signature → action → displacement`. It is `(contact with tile-class T) →
(attribute-delta on a named remote object)`. Getting this from the probe means the
0016 slot can be the right SHAPE on the first try.

## Gap B — scale-invariant equality: CONFIRMED REQUIRED
Measured: A areas {24,34}, B target area ~6 (B also showed 16/21 = occlusion noise
when the avatar nears the corridor). A ≈ 4× B (2× linear). Every 0010 shape lens
(raw/translation/rotation/reflection/rotation+colour) is **area-preserving**, so it
can NEVER equate masks of different area → `equal(shape,A,B,under)` is False for all
existing lenses. Computationally: A==B under existing lenses = False; under
scale-normalized shape (rot/refl) = True. A and B are the same motif at different
scale. The win-equality needs a SCALE lens 0010 lacks. (This is the trap caught
before writing code: calling equality "done" would have shipped a scale-blind
predicate — `match_fire=0` one level up.)

## The three gaps, ranked and resolved
- Gap A (remote cause→effect): REAL, load-bearing. Trigger = contact-with-tile.
- Gap B (scale lens): REQUIRED, load-bearing (proven by area mismatch + scale-match).
- Gap C (cross-object equality wiring, sprite-vs-sprite): genuinely just wiring, but
  blocked on B — it must be scale-aware.

## What 0016 now is (specified by measurement, not guessed)
1. A CHANGER-at-a-distance slot: `contact(avatar, tile-class T) → attribute-change
   of object Y`, recording WHICH object changed (avatar vs remote) — generalise
   0009's egocentric `changed?` over all objects.
2. Add a scale lens to 0010's invariance lattice.
3. Scale-aware cross-object equality A vs B, tied to the reward channel; the goal
   becomes "drive workspace A (via the cross) to scale-match invariant template B,
   then enter B's chamber" — guided, no blind sweep. Stated over discovered
   controllability; never names cross/box/shape/door.

## Honest residual (the next falsifier)
Not yet won, so the engine hasn't confirmed "A scale-matches B (+ on door) ⇒ level
tick." The scale-norm match could be partly 8×8-resample coincidence. With the
line-378 fix (reward now read on terminal frames), the clean test is: hammer the
cross to drive A to each rotation, walk into B's chamber, watch for the tick.
