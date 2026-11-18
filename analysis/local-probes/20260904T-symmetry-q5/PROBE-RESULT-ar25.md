# ar25 offline separator — pair exists; it does not mark the terminal

2026-09-04. Zero tokens. Seeded `symmetry.py` (`7e0c1f1c…`).
Frames: `analysis/matched-actor-v5-pilot/checkpoints/symmetry-uptake-ar25/level_0/turn_*.txt`
(17 grids, turns 0–16). The live uptake run dir was untracked; this checkpoint is
the surviving public dump. Script: `probe_ar25.py`.

Uptake actor (turn 3): `congruent(fig4, fig11, translate=True)` with labels
flattened, sizes 45==45, then 10 DOWN + 4 LEFT, level complete on LEFT.

## The pair

On turn 0, excluding background 9, barrier 10, HUD row 63 and move-counter
column 63:

| piece | colour | n | origin | shape |
|---|---:|---:|---|---|
| mover | 4 | 45 | (15, 36) | L, 9×9 bbox, 3-wide stem |
| target | 11 | 45 | (45, 51) | the same L |
| other | 5 | 40 | — | different (holes); never congruent to 4 |

## Relations, every L0 turn

`congruent(..., translate=True, colours=any)` and `directly_congruent` and
`similar` and nonempty `transporter` are **true on turns 0–14**, false on 15–16.

`colours=fixed` is **never** true (4 ≠ 11). The actor's flattening was required.

`translate=False` (same place) and `same origin` are **never** true. Overlay
fragments the CCs: at turn 15, colour 11 splits 18+1+18 and colour 4 drops to 44;
at 16, 43 vs 18. The win is not two intact figures occupying one origin. It is
one piece covering the other until the extraction no longer sees two L's.

## §3 as written for ls20

“True at the terminal, false on the others.” **No relation here does that.**
Shape congruence is a standing fact *before* the win and dies *at* the win.

## What this licenses

A this-versus-that pair **exists on ar25 seed 0 level 0**, recoverable from
public frames with the seeded module: colour-4 L ≡ colour-11 L under
`translate=True, colours=any`. That is step (2) of the chain — two figures,
one quotient, `True`. It is not a terminal signature, and it does not say
that matching wins. The actor still has to bet that transporting one onto
the other completes the level.

ls20 had no such pair. ar25 has one. That is the difference the separator
was for.

Colour-4 vs colour-5 is false every turn (negative control).
