# Hypotheses

## Color map (confirmed)
2=dark-gray frame 3=green/olive 4=light-gray empty-fill 5=bg 11(B)=yellow 13(D)=red 14(E)=light-green

## H2: bottom bar (row 63, 64x color3) drains from the RIGHT per click, 3->4
- status: CONFIRMED draining (right-to-left, 3->4). Consumption: click1=1, click2=2 (cumulative 1,3).
- sub-hypotheses for cost:
  - H2a triangular/time: cost = click# (1,2,3,...) -> cumulative 1,3,6,10,... (64 cells ~ 10-11 clicks)
  - H2b per-object cost: greenbox=1, yellowbar=2
  - H2c constant per click (but contradicted: 1 then 2)
  - DECIDING TEST: click an EMPTY background cell.
    * drains 3 => H2a (global clock, triangular)
    * drains 1/2 => global clock w/ other cost rule
    * no drain => bar only tracks object interactions (H2b territory)

## H1: clicking a piece's red dot moves/rotates the piece
- status: CONTRADICTED so far — 2 dot clicks, zero piece movement, no animation frames.
- alternative: click = SELECT piece (state change invisible except via bar), second click = destination?
- test after cost disambiguation: click green dot then click green container / diamond.

## H3: red diamonds (centers (10,52) & (52,10)) = targets / hazards / clickable
- untested. Aligned with piece dots: box dot (10,31) shares row 10 with (10,52); bar dot (34,10) shares col 10 with (52,10).

## H4: containers receive pieces (green Cs horizontal @ rows18-24 cols36-48; yellow caps vertical @ rows35-47 cols21-27)
- piece color matches bracket color (box 14 vs Cs 14? note: box fill=14, Cs=14; bar 11, caps 11).
- untested.

## Geometry (initial, col,row 0-indexed)
- green box rows9-11 cols27-32, dot (row10,col31), left edge col27=3
- yellow bar rows27-35 cols9-11, top row27=3, dot (row34,col10)
- diamond A center (10,52); diamond B center (52,10)  [4-cell plus shapes]
- horiz container frame rows18/24 cols36-48 (2), fill 4, divider col42 rows19-23 (3), Cs (14)
- vert container frame cols21/27 rows35-47 (2), fill 4, divider row41 cols22-26 (3), caps (11)
- bottom bar row63: 64x 3 (start)
