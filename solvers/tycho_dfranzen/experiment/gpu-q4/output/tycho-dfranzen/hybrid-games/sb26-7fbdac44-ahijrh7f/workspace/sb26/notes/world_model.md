# world_model.md — notes for the Builder (owner of world_model.py)

## Game family (evidence: level 0 solved, level 1 started)
"Sequence panel": a **target strip** of N hollow coloured boxes along the top; a **panel** (or
several) of empty slots; a **tray** of N swatches in shuffled order. Win = reproduce the strip's
left-to-right colour sequence in the slots, then **submit with SPACEBAR**.

## State to encode (keep it cheap — the grid is 64×64, colours 0–15)
- `target = [colour of strip box i]` read from row 1 of the top strip (boxes are 5–6 cells wide,
  separated by `5`/black columns). Level 0: `[9,14,11,15]`; level 1: `[12,15,8,9,14,11,6]`.
- `slots = [(r0,r1,c0,c1,fill)]` — empty slot interior is a 2×2 of `2`; a filled slot is a solid
  4×4 block (rows r−1..r+2 of the interior) of the deposited colour.
- `tray = [(r,c,colour,used)]` — 2×2 (level 0) / 3×4–4×4 (level 1) blocks in the bottom band
  (rows 57–60), colour `2` once spent.
- `selected` = the swatch currently ringed by a white `0` outline (level 0: outline occupies
  rows 56–61, cols c−3..c+2 around a 2×2 swatch).
- `timer` = count of `2` cells in row 53 (starts 64, decrements ~1 per deposit).

## Action laws (verified on level 0, 11 transitions + terminal)
- `CLICK(swatch)` → `selected = swatch`; draws white ring; timer unchanged.
- `CLICK(empty slot interior)` with `selected = s` → slot becomes solid `s`; `used(s) = True`;
  `selected = None`; timer −1.
- `CLICK(anything else)` → no-op (safe, but still burns a budgeted action — avoid).
- `UNDO` → inverse of the last deposit (restores swatch colour and empty slot).
- `SPACEBAR` → **evaluate**. If slots == target (left-to-right) → `win`; otherwise unknown
  (never observed a wrong-submit; may be a no-op or a loss — do not assume harmless).
- Idempotence: re-clicking the same swatch just re-selects it. A filled slot cannot accept a
  second deposit (not observed; treat as unverified).

## Objective / verification key
`slots.fill == target` **and** last action was SPACEBAR. Filling all slots is a *subgoal*, not
the goal — level 0 sat fully correct for one whole turn and only the SPACEBAR won it. Any planner
key that merges "panel complete, unsubmitted" with "panel complete, submitted" is wrong; keep
`last_action` (or a `submitted` flag) in the key.

## Budget — the real constraint
`action_limit = 16` for the **whole episode**, shared across levels (level 0 used 12). A correct
plan must be near-minimal: 2 CLICKs per slot (select+deposit) batched into one 4-action queue,
then SPACEBAR. Do not model exploratory clicks as free.

## Level 1 open hypotheses (distinguish before planning)
1. The small hollow green box (rows 21–24, cols 33–36) + 2-wide green stem (rows 25–31, cols
   34–35) + second green-framed panel (rows 32–41) may mean: (a) two independent panels that
   together hold the 7-colour sequence (3 in red + 4 in green); (b) the green box is a hopper —
   deposits into it are *routed* into the green panel; (c) the green panel is an output/echo that
   fills automatically.
2. Whether one CLICK can fill more than one slot (batch/pipe behaviour) — if so the budget is
   survivable.
Cheapest discriminating probe: `CLICK(orange swatch 58,31)` then `CLICK(red slot 1, 22,22)` and
read where the colour lands (red slot only, green panel too, or the green box).
