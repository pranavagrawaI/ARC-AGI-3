## Turn 0 — initial inspection
Grid colors: 0=bg, 1=tile, 5=frame, 9=HUD row, 10=offboard, 14=gem/cross (red).
Structure: framed playfield rows 10-49. Left region rows 10-28 (21x44), 6x3 slot
grid (tiles 4x4, gaps 2). Right region rows 30-49 (20x15), 3x2 slot grid.
8 gems (14) on tiles. Band of 9 under frame at row 29 (cols 10-38) and row 50
(cols 37-53) — possibly HUD/level info.
H4 (rigid whole-blob shift) predicts k=0 for ALL directions — blob touches
frame everywhere, so all moves would be no_impact. Need to test another
mechanic.
Next experiment: try RIGHT, then DOWN etc., observe diffs.

