# World model notes — Level 0 (start)

## Static scene (level 0, turn 0)
- Palette used: 0 white, 1 light-grey, 3 dark-grey, 4 charcoal, 5 black, 8 red, 9 blue, 11 yellow, 12 orange.
- Black vertical strip cols 0-3 (rows 0-51) = frame.
- Top box: rows 8-16, cols 32-40: dark-grey frame (3), black interior (5), small blue glyph
  ("7"-shape) at (11,36)(11,37)(12,37)(13,36)(13,38).
- Black pillar cols 34-38, rows 17-24.
- Big dark-grey region rows 25-49, cols 14-55 ("room"), containing:
  - white glyph at (31,21),(32,21),(32,22) + light-grey (32,20),(33,21) — arrow/"7"-like, candidate PLAYER.
  - charcoal (4) pockets: rows 30-39 cols 31-35 (vertical), rows 40-44 cols 10-18 & 24-33,
    rows 25-29 cols 14-55 (top band), rows 45-49 right half.
  - orange (12) 5x2 at rows 45-46, cols 38-42; blue (9) 5x3 at rows 47-49, cols 38-42.
- Field above (rows 0-24) mostly charcoal 4 (open?), black logo box bottom-left (53-62, 1-11)
  with big blue "6" (rows 55-59, cols 3-9) = likely level-number decoration.
- HUD rows 61-62: yellow bar (11) cols 13-54; red+black dashes cols 56-63 (decor?).

## Hypotheses
H1 (leading): 4=walkable floor, 3=wall/floor-of-room-unknown; player=white glyph; move with
ACTION1-4; goal = deliver blue to blue-box (top) / orange is a block to push.
H2: 3=floor, 4=wall; player inside dark room, pockets are holes; blue box = goal.
Distinguishing probe: ACTION1 — watch where the white glyph moves and what it bumps into.

## OBSERVED t0→t1 (after ACTION1="up")
- White glyph at (31-33,20-22) did NOT move.
- Block (orange 12 top 2 rows + blue 9 bottom 3 rows, 5 wide x 5 tall) at cols 34-38
  translated UP 5 rows: rows 45-49 -> rows 40-44. It snapped into a 5-row slot.
- Yellow bar (11) rows61-62 shrank by 1 column (col13: 11->3): 84->82 cells. = 1 unit spent.
  => budget ~42 units (2 cells per unit, bar is 2 rows tall).
- Block is 5x5 macro-cell; room rows 25-49 = 5 slots of 5 rows. Hypothesis: 5x5-slot grid.
- Room has a 4-shaft at cols 30-34 rows 30-39 (vertical gap in the 3-floor).

## REVISED identity
- Block (orange+blue) = moving player/object. White arrow glyph = marker/goal (TBD).
- 3 = floor (block can occupy & moved within 3). Movement = 5-row slot step.
- Need: does each action = one 5-slot step? does it slide-to-wall instead?
- Need: what ends the level (goal = reach white arrow? reach top box? clear bar?).

## OBSERVED t1→t2 (2nd ACTION1)
- Block moved up again exactly 5 rows: rows 40-44 -> 35-39. Budget 82->80 cells (41 units).
- => Each directional action = 1 slot (5 px) move, always succeeded so far; budget 42 units start.

## Block column (col 34-38) vertical map (current):
- slot rows 35-39: BLOCK (top=orange 35-36, bottom=blue 37-39)
- rows 25-39: all floor(3) in col 34-38 (open)
- rows 17-24: floor 4 at cols 32-33 & 39; OBSTACLE(3) only at cols 34-38 (the "pillar")
- rows 8-16: TOP BOX (frame 3, interior 5, blue "7" glyph 11-13) — matches block's BLUE part
- rows 0-7: open(4)
Path to box = straight up 3 more slots. Box blue glyph strongly suggests GOAL (blue->blue match).

## White glyph (31-33,20-22) = static marker, likely directional hint (looks like up-ish arrow)
or a slot marker. Not a wall (block moved freely over row 31-39 col 34-38).

## OBSERVED t2->t3 (ACTION3 = left)
- Block did NOT move (orange still rows 35-36, cols 34-38). Only HUD bar shrank (col15 11->3).
- => Blocked by 4-pocket at cols 32-33 (rows 30-39). CONFIRMS: 3=floor, 4=WALL, 5=wall.
- Blocked moves still cost 1 budget unit. Budget now 39 units (bar cols 16-54).

## Path analysis (block rows 35-39, cols 34-38)
- Up: rows 30-34 (all 3) OK; rows 25-29 (all 3) OK; rows 20-24 (pillar, all 3) OK;
  rows 15-19 blocked (row15 cols34-38 = 5, box interior). Max slot = rows 20-24 (under box).
- Goal hypothesis: terminal when block reaches slot rows 20-24 cols 34-38 (directly under
  top box / on pillar top), OR when any top-cell of block touches box frame row 16.
  Fallback: white glyph slot (rows 30-34, cols 20-24) — but left path is walled (4 pockets
  at cols 32-33 and 24-33 region) so likely NOT it.

## PLAN
1) ACTION1 x3 — up to rows 20-24. Watch for terminal at 30-34 / 25-29 / 20-24.
2) If no terminal at 20-24, probe ACTION1 once more (expect blocked no-op), then consider
   ACTION2 down / lateral detours around 4-walls, or that the white glyph is a target to
   push/align differently.
Alternatives if goal != box: white-glyph slot alignment; orange->? (no orange target exists).
- Which color is floor vs wall? Is 5 black also obstacle?
- Does the glyph rotate/teleport? Does pushing exist (orange under blue)?
- What ends the level (terminal grid)?
- Are rows 61-62 / red dashes live HUD (lives/step counter)?

## OBSERVED t6->t7 (ACTION1 up)
- Block (orange top 2 rows, blue bottom 3 rows, cols 34-38) moved up 5 rows: rows 25-29 -> 15-19.
- Block NOW OVERLAPS the top box: row 15 (box interior, was 5/black) -> orange; row 16
  (box bottom frame, was 3) -> orange. Box frame is NOT an obstacle; block passes through.
- Box interior rows 9-15 cols 33-39; blue glyph at rows 11-13 cols 35-38 (static, untouched).
- Still NOT_FINISHED with block top inside box. Goal not "touch box".
- Budget bar now cols 20-54 = 35 units left (was 42).
- Block top-row history: 45,40,35,30,25,20,15 -> top row always multiple of 5.
  Possible tops: 10, 5, 0 (or blocked at screen edge).
## GOAL HYPOTHESES (ranked, probe = keep pressing up)
G1: terminal when blue part overlaps/enters box glyph rows 11-13 -> next move (10-14) blue=12-14
    overlaps row 12 of glyph. Check after t8.
G2: terminal when block fully inside box (top 10) or reaches top edge (top 0-4).
G3: box is a portal: block exits top (rows 0-4) then level ends.
Next: ACTION1 (top->10). If ongoing, ACTION1 (top->5). If ongoing, ACTION1 (top->0 or blocked).
If all fail: try ACTION3/ACTION4 lateral inside box, or ACTION2 back down; reconsider glyph as
match-target (blue cells must equal glyph cells exactly?).
