# Friction stack on F1x: no build (2026-10-18)

The proposed stack was F1x (preload + 16K output) + a wording fix for B1's animated no-op contradiction +
possibly `ARC3_PRIORITY_PACE=1`. Neither addition survives evidence, so the stack is F1x alone, which is
already queued as D10 (2026-10-19). No screen is spent.

## Animated no-op wording: no mechanism

`contradiction_followups.py` over the four B1-family 60-min screen arms (screen-1 f1/b1ctl, screen-2
f1x/b1ctl; 100 game logs) → `followups.json`:

| turn after a no-op guard stop | n | read the animation | known-no-op BLOCK / StaleStateActionError in the turn |
|---|---:|---:|---:|
| animated stop (message says both "not a no-op" and NO-OP) | 20 | 13 | 0 |
| plain stop | 57 | 3 | 2 |

The contradiction occurs about 5 times per 60-min run. After it, the model does what the animation line
asks: it reads `last_animation_timeline`/`frames` (13/20) and never re-executes the stopped action into the
known-no-op block (0/20). There is no wasted action or turn for a rewording to remove. The `repeat` regex
only sees literal `action([...])` lists, so the BLOCK count is the reliable waste signal: B1 blocks
any re-execution of a known no-op in the same board state.

## ARC3_PRIORITY_PACE: in a class already measured null

PACE scales a game's priority by its token cost against a per-level reference. That is a token
reallocation between games, the class the 10-13 scheduler simulator measured at ≈0 or harmful (FIFO −5%,
park-after-T ±1%, retuned token scale ≈0; analysis/scheduler-decision-20261016/NOTE.md), and the
workstream was closed unless a mechanism outside that class showed up. PACE's only support is
`jingshuifu/arc3-m2-solution-v3` 32.60 (n=1, +1.5 sd over the public dfranzen mean 28.01 ± 3.03). The one
other public draw with PACE, `themightiestman/arc3-yukailiu-dprime` (dprime+pace+reset+temp0.70), scored
28.24. Not admitted.
