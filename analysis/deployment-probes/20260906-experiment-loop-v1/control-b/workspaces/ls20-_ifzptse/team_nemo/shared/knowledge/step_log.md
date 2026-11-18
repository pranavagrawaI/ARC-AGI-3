## Turn 0 (step 0, level 0) — experiment 1: UP
Grid at start: 64x64. Colors {4,5} fill most; a 3-bordered "blob" region rows 8-51, cols ~13-43 with interior 0s/1s;
bottom-left cluster rows 52-62 cols 3-7 (5/9 block); a b-block rows 24-26 cols 55-63; 8s near row 24 col ~58.
Rows 0-7 entirely {4,5}; ROI rows0-3 cols60-63 all 4.
**Tested hypothesis:** UP changes pixels in rows 0-3, cols 60-63.
**Alternative:** unchanged.
Submitted: [UP] with predicted_path move.
Pre-grid saved as g0; ROI pre = all 4s.

## Turn 1 (step 1, level 0) — UP result: hypothesis CONTRADICTED
Submitted UP at step 0. action_results: UP -> move, diff_pixels=52, label=playfield.
**Prediction scored: move = move ✔ (batching may reopen if I predict correctly).**

Diff analysis (pre g0 from traj[0], post g1 from traj[1]):
- ROI rows0-3 cols60-63: **unchanged** → provisional hypothesis refuted.
- Changes:
  (a) A 5x5 patch at cols 34-38: rows 40-41 (was 3, now 12), rows 42-44 (was 3, now 9),
      rows 45-49 (was 12/12/9/9/9, now 3). Interpretation: a vertical column of cells
      in cols 34-38 **shifted up by 4 rows** (contents rolled upward, wrapped/culled at top).
      The 5-wide block rows 40-49 of [12,12,9,9,9,3,3,3,3,3] became rows 36-45 of
      [12,12,9,9,9] followed by 3s — i.e. the pattern moved UP 4 rows and the bottom
      4 rows filled with the surrounding border color 3.
  (b) (61,13): 11->3 and (62,13): 11->3 — two cells in the bottom status band lost color 11.

So UP moved a 5-wide object upward by 4 rows, and altered two 11-cells near the bottom-left.
The "object" is likely a 5x5 block of 12s (top) + 9s sitting at rows 40-44 cols 34-38 (pre),
which after UP is at rows 36-40 (wait — pre rows 40-41=12, 42-44=9 => 3 rows of 12 + 3 rows of 9 = 6 rows of object; post rows 40-41=12, 42-44=9 => same 5 rows but shifted up 4 rows, and the 3s that were at 45-49 are now 49+).

Actually re-examining: pre rows 45-46=12, rows 47-49=9 (below the 3-block). So the pre
"3-block" of 5x5 (rows 40-44) is 3s and rows 45-46=12 (2 rows), rows 47-49=9 (3 rows).
Post: rows 40-41=12 (2 rows), rows 42-44=9 (3 rows), rows 45-49=3. So the 12s moved from
rows 45-46 to 40-41 (up 4), and 9s from 47-49 to 42-44 (up 3?). Not a pure shift.
Maybe: the 12s and 9s are two separate objects moving up, with different velocities?
Or the whole 5x5 at cols 34-38 rows 45-49 (12,12,9,9,9) shifted up 4 rows to rows 41-45,
but rows 41-42 now show 12,12 not 12,12? Wait post shows 40,41=12 and 42,43,44=9.
Hmm, let me not over-interpret; I'll record the exact diff and refine later.

Second experiment choice: the 11->3 change at (61,13),(62,13) is a status-band effect.
A good second test: submit LEFT (or DOWN) with a spatial prediction. DOWN is the natural
reverse of UP; I predict DOWN will shift the same 5-wide object back down (and may restore
the 11s). Prediction: DOWN -> move, and the 5x5 12/9 block moves back to rows 45-49
(pre-UP position), and the 11-cells at (61,13),(62,13) restore to 11.

