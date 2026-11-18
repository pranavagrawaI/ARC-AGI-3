# Hypotheses
- H1 (turn 0): UP changes pixels in top-right ROI rows0-3 cols60-63. **CONTRADICTED** (turn 1): ROI unchanged; changes are at cols 34-38 rows 40-49 and bottom (61-62,13).
- H2 (turn 1): UP shifted a 5-wide (cols 34-38) object upward; the 5x5 region at rows 45-49 (pre: 12,12,9,9,9) moved up 4 rows to rows 41-45; the two 11-cells at (61,13),(62,13) lost color 11 (11->3). **Refinement pending**: exact shift is not a uniform 4-row translation; see step_log.
- H3 (turn 1, to test): DOWN is the inverse of UP — the 5-wide object returns to its pre-UP position and the 11s at (61,13),(62,13) restore.

