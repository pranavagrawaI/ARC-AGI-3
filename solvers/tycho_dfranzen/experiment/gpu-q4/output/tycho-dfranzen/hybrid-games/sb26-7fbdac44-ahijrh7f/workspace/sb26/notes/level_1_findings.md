# Level 1 — observations (episode budget almost exhausted: 14/16 actions used)

## Board
- Target strip (rows 0–7): 7 hollow boxes, left→right `12(c), 15(f), 8, 9, 14(e), 11(b), 6`.
- Tray (rows 57–60), 4×4 swatches, shuffled: `8`@cols 9–12, `15`@16–19, `14`@23–26, `12`@30–33,
  `6`@37–40, `9`@44–47, `11`@51–54.
- **Two panels**: red frame `8` (rows 17–27, cols 21–46, white corner brackets) with **3** empty
  slots (2×2 of `2` at rows 22–23, cols 22–23 / 28–29 / 40–41); green frame `14` (rows 32–41,
  cols 18–49) with **4** slots (rows 36–37, cols 22–23 / 28–29 / 34–35 / 40–41).
  3 + 4 = 7 slots = 7 colours ⇒ the small hollow green box (rows 21–24, cols 33–36) plus its
  2-wide stem (rows 25–31, cols 34–35, passing through a gap in the red frame's bottom edge) is
  **not** a slot; it is a decorative conduit/coupling between the two panels.
- Timer row 53 now 63×`2` + one `3` at col 63 (one deposit spent).

## Transitions observed (2 actions)
1. `CLICK(58,31)` on the orange(12) swatch → white `0` ring at rows 56–61, cols 29–34. No timer
   cost. **Identical to level 0.**
2. `CLICK(22,22)` → slot interior rows 22–23 cols 22–23 became `12`, ring 4×4 (rows 21–24,
   cols 21–24); swatch reverted to `2`/background; selection ring cleared; timer −1.
   **Deposit law transfers unchanged to level 1, including into the red panel.**

## Reading of the goal
Fill red slots left→right with `12, 15, 8` and green slots left→right with `9, 14, 11, 6`
(strip order read continuously across the two panels), then **SPACEBAR to submit** (level-0 lesson:
filling is necessary but not sufficient).

## Budget lesson (the binding constraint, not the timer)
`action_limit = 16` is **global across levels**: level 0 consumed 12, leaving 4 for level 1, which
needs 7 deposits × 2 CLICKs + 1 SPACEBAR = 15. Level 1 is unwinnable this episode.
⇒ Next attempt: never spend more than the minimum per level (2 CLICKs/slot, batch 4 queued CLICKs
per decision, no exploratory no-ops, no UNDO, and consider whether RESET refreshes the budget).
Level 0's 12 actions included 3 wasted (a wrong-order deposit, an UNDO, and idle probes); the
minimum for level 0 was 9. Cutting that to the bone is what makes later levels reachable.

## RESOLVED: premature SPACEBAR is harmless (action 15, level 1, 1/7 slots filled)
`SPACEBAR` changed exactly **one cell**: timer row 53 col 62 `2→3`. State stayed `NOT_FINISHED`;
no slot, swatch, ring or strip cell changed; no loss, no terminal event.
⇒ **Submit law:** `SPACEBAR` always evaluates and costs exactly one timer unit; a wrong/incomplete
panel is simply not accepted — it is *not* punished. So submitting early is a safe probe, and
"panel correct but unsubmitted" is a distinct state from "submitted" (level-0 win) — keep
`submitted/last_action` in any planner key. Timer, not action count, is the in-level budget
(64 units; ~1 per deposit and per submit, so ~7 deposits + 1 submit ≈ 8 units — very generous).

## Final action (16th) reserved for: `RESET`
Never-observed semantics, and the only lever that could refresh the global 16-action budget.
Level-1 progress (1 orange deposit) is worthless with 0 actions to follow, so the probe is free.
Watch: does RESET restore level 1's initial frame (7 intact swatches, empty slots, timer 64)?
Does `level`/`actions_used`/`levels_completed` change? Does a new attempt begin with a fresh budget?
