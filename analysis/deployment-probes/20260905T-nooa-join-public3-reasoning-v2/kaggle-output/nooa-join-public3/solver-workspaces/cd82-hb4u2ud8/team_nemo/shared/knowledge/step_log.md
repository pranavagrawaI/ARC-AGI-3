## Turn 0 (step 0, level 0)
First state analyzed. Full element map verified to 0 mismatch pixels (see worldmodel.expected_level0):
- bg=5 purple; bottom border row 63 navy(4)
- top-left (r0-17,c0-17): A=black(0) 10x5 @r3-7,c3-12; B=white(15) 10x5 @r8-12,c3-12; navy wall c16-17 r0-15; navy base r16-17 c0-17
- top-right (r0-8,c18-63): solid olive(3); C=navy-framed black 3x3 @r2-6,c35-39; D=navy-framed white 3x3 @r2-6,c41-45; E=black strip 5x1 @r7,c41-45
- middle: F=green(2) frame box r24-32 c25-38 with white(15) fill r25-31 c26-37, OPEN bottom (r32 gap c26-37)
- bottom: G=black(0) 10x10 @r34-43,c27-36
No visible small player sprite — all non-bg pixels accounted for.
Actions: RESET, UP, DOWN, LEFT, RIGHT, SPACEBAR, CLICK.
Plan: probe with single actions to find the player / mechanics. First probe: UP.

## Turn 0b (still step 0)
- The first-state frame arrived on the queue only now; UP probe submitted on the previous handle is
  the registered submission for turn 0 (submitted_this_turn=True) and is PENDING (not yet executed:
  action_results empty, diff "first state").
- Prediction on file: UP -> "move" (grid changes). If it comes back no_impact, batching closes and I
  re-probe with LEFT/RIGHT/CLICK/SPACEBAR one at a time.
- Next-turn checklist:
  1. parse step-1 state; diff vs worldmodel.expected_level0() -> which element moved?
  2. locate the player: any new color / displaced component (encode['player']).
  3. if arrows inert -> probe CLICK (center of G ~ c31,r38) and SPACEBAR.

## Turn 1 (step 1) — UP probe
- UP executed. Harness outcome: "move" (prediction CORRECT, batching stays open).
- Grid diff vs baseline: EXACTLY 1 pixel -> (63,63): 4 -> 5. No playfield change. No animation frames.
- Full component scan: still the 15 static components. No sprite. A 5-colored (bg) player would be
  visible on any non-5 surface; none exists -> no visible player.
- Working model: (63,63) = step counter (4 + n_actions). Arrows are playfield no-ops (for now).
- Next probe: LEFT, RIGHT (batch, allowed since last retrodiction was right).
  counter hyp: (63,63) 5->6->7. player hyp: swap on border row, net-zero on 2nd.

## Turn 2 decision (batch LEFT, RIGHT, SPACEBAR)
- Rationale: disambiguate counter vs invisible-border-player; first-look at SPACEBAR.
- Expected if counter: (63,63) = 6, 7, 8 at steps 2,3,4; playfield static.
- Expected if border-player: step1 diff (63,62)4->5 + (63,63)5->4; step2 reverts (no_impact); step3 unknown.
- If all three are counter-only: next turn probe CLICK (plain first; try "CLICK x y" coords if inert).

