# Level 0 — solved (win at turn 11 by SPACEBAR)

## What the board was (colours are the numeric grid values)
- `4` background, `5` black, `0` white, `8` red, `2` grey (empty/depleted), `3` darker grey (spent timer).
- Palette seen: `9` blue, `14(e)` green, `11(b)` yellow, `15(f)` purple.
- **Top strip** (rows 0–7, cols 17–45): four hollow boxes, outline colours left→right
  `9, 14, 11, 15`. This is the **target sequence**.
- **Panel** (red frame `8`, rows 25–34, cols 19–45, white `0` corner brackets): four slots,
  interiors 2×2 of `2` at rows 29–30, cols 22–23 / 28–29 / 34–35 / 40–41.
- **Tray** (rows 57–60): four 2×2 swatches at cols 18–19 (`14`), 24–25 (`15`), 30–31 (`9`),
  36–37 (`11`) — **shuffled relative to the target order**.
- **Timer bar** row 53: 64 cells of `2`, consumed right-to-left (`2→3`), ~1 unit per deposit-like
  action. 58 units left at the win → generous budget, not the binding constraint.

## Mechanics (all verified against observed transitions)
1. `CLICK` on a tray swatch → a white (`0`) 6×4 selection outline appears at rows 56–61 around it.
   Selection costs no timer.
2. `CLICK` on an **empty slot interior** while a swatch is selected → the slot becomes a solid
   4×4 block of that colour (rows 28–31, cols X−1..X+2 for interior cols X..X+1); the swatch
   turns to `2` (spent) and its selection box vanishes; timer loses 1 unit.
3. Deposits are one-shot: the swatch is consumed (no re-use, no colour left in the tray).
4. `UNDO` reverted a deposit (slot back to `2`, swatch colour restored).
5. `CLICK` on background/empty space = no-op.
6. **`SPACEBAR` = submit.** At turn 11 the panel already read `9,14,11,15` (matching the strip)
   but state stayed `NOT_FINISHED`; the single SPACEBAR produced `outcome: win`. In the terminal
   frame the only change attributable to the submit is one more timer cell spent — the target
   boxes are drawn **filled** in the terminal frame (they were hollow outlines while playing),
   i.e. the win frame "activates" the strip.

## Goal, distilled
Reproduce the top strip's left-to-right colour sequence in the panel's slots, one swatch per
slot (tray order is a permutation of the target order, so read the strip, do not click in tray
order), then **press SPACEBAR to submit**. Filling all slots is *not* sufficient — submission is
a separate, required action. That is the single most transferable lesson.

## Strategy that worked
select swatch → deposit → repeat, always taking the colour the strip demands for the next slot;
batch select+deposit as two queued CLICKs (guards on the public successor held every time).
Total 8 CLICKs for 4 slots + 1 SPACEBAR.

## Expected next-level variation (level 1, observed)
- Target strip now has **7** boxes: `12(c), 15(f), 8, 9, 14(e), 11(b), 6`; new colours `12`
  (orange), `8` (red), `6` (pink). Tray now has **7** swatches in shuffled order
  `8, 15, 14, 12, 6, 9, 11` at rows 57–60, cols 9–12 / 16–19 / 23–26 / 30–33 / 37–40 / 44–47 / 51–54.
- The play area is now **two framed panels**: a red frame (rows 17–27) and a green frame
  (rows 32–41, colour `14`), joined by a green 2-wide **stem** at cols 34–35 that passes through
  a gap in the red frame's bottom edge, plus a small hollow green box at rows 21–24, cols 33–36
  sitting in the red panel's third slot position. Red panel shows grey slot pairs at cols
  22–23, 28–29, 40–41 (3 visible); green panel at cols 22–23, 28–29, 34–35, 40–41 (4).
  Open question: is the green box a *slot*, a *hopper/cursor*, or just the stem's mouth?
  7 colours vs 3+4 slots suggests the green box is not a slot.
- **Action budget is global, not per level**: 16 for the whole episode; level 0 consumed 12, so
  only 4 remain for level 1 → level 1 cannot be completed by pure deposit-per-slot play.
  Future runs must be far more economical: batch 4 queued CLICKs per decision, no exploratory
  no-ops, no UNDO.
