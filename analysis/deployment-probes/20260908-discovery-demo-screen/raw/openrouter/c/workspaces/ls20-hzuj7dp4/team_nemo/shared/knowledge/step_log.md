## Turn 0 - initial inspection
Grid 64x64. Colors: 4=bg, 3=wall (maze corridors), 5=frame/border, 9=trail/orange, 12=c(12), 0=white, 1=red, b=bottom band, 8=corner box.
Layout:
- Player dot: (31,20)=0, (32,21)=0 white; 1 at (32,20) and (33,21). Likely a 2x2 player sprite (0,0 / 1,0 / 0,1 style).
- 3-walls form maze rooms: top room rows 9-15 cols 33-39 (with 9 pixels inside: (11,35-37),(12,36? actually (12,36)=9? row12: 355559553 -> col 37=9), (13,35),(13,37)); middle corridor rows 17-24 cols 34-38; big lower area rows 25-49; side rooms rows 40-44 cols 19-23; c-block rows 45-46 cols 34-38 (12); 9-block rows 47-49 cols 34-38.
- Bottom band rows 61-62 cols 13-53 = b; rows 61-62 cols 56-63 = 8 (with 5 in col 55).
- 9-trail: rows 55-59 cols 7-10ish: (55,6-9),(56,6-9),(57,6-7),(58,6-7),(59,6-7),(59,9),(59,10).
Hypotheses:
H1: 12 (c) = target/goal. H2: 9 = player trail (path walked), 9-block at rows 47-49 = another goal or item.
Plan: try UP/LEFT to see which cell the sprite moves, whether trail 9 is left behind.

