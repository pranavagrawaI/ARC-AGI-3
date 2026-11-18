# Level 0 — World Model Notes (recomputed, rule VERIFIED 1.00 / 10-11 transitions)

## Mechanic (CONFIRMED, model accuracy 1.00)
- Only ACTION6 (click) + RESET exist. Only row 63 ever changes.
- Row 63 is a track of value 3; fill of value 4 grows LEFT from col 63.
- Each ACTION6 click grows the fill:
    inc = 2 if L % 9 in {1, 6} else 1     (L = PRE-click bar length)
  +2 milestones (preL): 1,6,10,15,19,24,28,33,37,42,46,51,55,60,(64)
  (+2 click positions: 2,6,9,13,16,20,23,27,30,34,37,41,44,48,51,55,58,62)
- Target cell is IRRELEVANT (16 varied clicks, delta depends only on L).
- The 7 objects are static; no non-row-63 cell ever changed under a click.
- (Earlier mod-5 rule was WRONG; +2 preL set was {1,6,10}, not a mod-5 class.
   Correct fit: L%9 in {1,6}, matches all 16 observed clicks 100%.)

## Full click log (preL -> delta) — 2 attempts, all match rule
Archive attempt 0: 0->+1,1->+2,3->+1,4->+1,5->+1,6->+2,8->+1  (reset at L=9)
Current attempt:   0->+1,1->+2,3->+1,4->+1,5->+1,6->+2,8->+1,9->+1,10->+2,12->+1
  => now L=13 (bar cols 51..63) at turn 10.

## State now (turn 10): bar length 13.
Next click: preL=13 -> 13%9=4 -> +1 -> L=14.

## Goal analysis (STILL UNKNOWN — no terminal seen)
- No terminal at L=9 or L=13. Bar fills monotonically, target irrelevant.
- Top hypothesis: fill bar to FULL = 64 -> reached at click #50 (L:13 ->
  14,15,17,18,19,21,22,23,24,26,27,28,30,31,32,33,35,36,37,39,40,41,42,44,
  45,46,48,49,50,51,53,54,55,57,58,59,60,62,63,64).
  From L=13 to 64 = 51 more clicks.
- Competing target candidates (all just "enough clicks"): 15,20,32,64.
  Indistinguishable until the bar hits one; monitor after every click.
- Risk: possible unseen goal / penalty; no evidence of any (only the bar
  ever moves). RESET is available and is a strategy control, not a death.

## Plan
- Click (0,0) each turn (safe bg, delta is target-independent).
- Watch for level_complete at any bar length; keep going to 64 if none.
- No-ops would indicate a target < 64 that was overshot — none observed.
