
## h0 (initial read of level 0, step 0)
- World: 64x64. Ground=4, walls/terrain=3 (upper plateau band rows ~25-39 and lower rows 40-49 near cols 14-20),
  player=5 block cols 31-37 rows 30-63 (vertical column of 5s; head at row 30, extends to row 63).
- Objects on plateau: 1/0 'dot' at rows 31-33 cols 21-22 (on 3); 12 'c' block cols 34-38 rows 45-46 (on 3);
  9-block cols 34-38 rows 47-49 (below it); 9-pixies in top object rows 11-13; 8-block cols 56-63 rows 61-62 (bottom right).
- Hypothesis A: player is a vertical 5-column (cols 31-37) standing on the 3-ground; movement slides it.
- Hypothesis B: 'c'(12) at 45-46/34-38 is a target item; 9 block below may be its pedestal; dot 1/0 at 31-33/21-22 a collectible;
  8-block bottom-right maybe decorative or another item.
- OPEN: which pixel is the 'agent' (which cell moves), what the objective is, what 3 vs 5 vs 4 mean.
Test plan: single LEFT probe — look at which cells change; the changed cell set tells us the agent's footprint.


## turn 1-2 — LEFT, LEFT (steps 1,2)
- Each LEFT translates the item (12-top/9-bottom 5x5 block, rows 45-49) exactly -5 cols. c0: 34->29->24.
- c-panel (rows 61-62, color 11=b) grows one col left per LEFT: c0 13->14. So c0 = 13 + 5*n(LEFT)? (13,14 after 1,2 LEFT... actually c0 increased: 13 then 14). Track separately.
- The 5x5 at cols26-32 rows30-63 (vertical bar) is STATIC.
- Model world_model.py encodes item(r0,c0), predict does +-5 per dir (horizontal confirmed; vertical untested).
Goal unknown. Item sits on a 3-floor (rows up to 49) over a 4-floor (row50+). Probe UP to learn vertical rule.


## step 4 RIGHT — measured
- 52 cells changed = 25 item-12 cells + 25 item-9 cells (both +5 cols) + 2 c-panel (61/62,16 b->3).
- item 12 before cols 24-28 rows 45-46; after cols 29-33 rows 45-46: exact +5 shift (verified set equality).
- item 9 before cols 24-28 rows 47-49; after cols 29-33 rows 47-49: exact +5 shift.
- c-panel loses exactly 1 col on EVERY action (LEFT, RIGHT, and blocked UP all shrink it by 1 col): it is a step/fuel counter, 1 col per action.
- UP at step 3 changed only the 2 c-panel cells (item didn't move): UP is blocked (no 3-ground above rows 40-44).
- 1/0 dot, 9-pixies, 5-column, 8-block: all static so far.
- Item now at rows 45-49, cols 29-33.

