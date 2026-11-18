## Turn 0 (step 0)
Initial inspection. Measured:
- Block: 5x5 at rows 45-49, cols 34-38. Top 2 rows=color 12, bottom 3 rows=color 9.
- Window: rows 40-49, cols 34-43. Block at bottom-left.
- Bar: rows 61-62, cols 20-54 (35 cells, color 11).
- Static elements: top box (rows 8-16, cols 32-40), 8-blocks (rows 61-62, cols 56-63), 0/1 pixels (rows 31-33, cols 20-22).
- Background: 3 (dark), 4 (mid), 5 (light) in large regions.

Hypothesis: Direction moves shift block by 5 in that direction, bouncing at window edges. Bar shrinks left edge by 1 per action.
Test: RIGHT should move block cols 34->39, bar left 20->21.

