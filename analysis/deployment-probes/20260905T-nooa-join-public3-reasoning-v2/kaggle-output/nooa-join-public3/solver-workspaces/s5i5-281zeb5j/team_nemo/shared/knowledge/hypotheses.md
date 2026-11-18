# Game knowledge

## Initial state (turn 0, level 0)
- Actions available: RESET, CLICK (coordinate-based game, no movement).
- Background = 5. Status band = row 63 (all 3s at start).
- Non-background objects (8-connected components):
  - O1 "small box" rows 9-11, cols 27-32: border = 3 on left col (27), 14 on other 3 sides,
    core 13 at (10,31).
  - O2 "diamond" 13 at rows 9-11 cols 51-53 (pluses shape, center (10,52)).
  - O3 "big frame A" rows 18-24, cols 36-48: border 2; interior: vertical 3-line col 42
    rows 19-23; 4 fills sides; two 14 "eyes" at rows 20-22 (3x3 minus center) centered
    (21,39),(21,45).
  - O4 "tube" rows 27-35, cols 9-11: top row 3, body 11, core 13 at (34,10).
  - O5 "big frame B" rows 35-47, cols 21-27: border 2; interior: horizontal 3-line row 41
    cols 22-26; 4 fills; 11 pattern rows 37-39 (3x3 ring) and rows 43-45 (3x3 block).
  - O6 "diamond" 13 at rows 51-53 cols 9-11 (center (52,10)).
- 13 appears as: cores inside O1, O4, and the two standalone diamonds O2, O6.
- 11 appears as: body of O4, and interior pattern in O5.

## Hypotheses
- H1: CLICK x y clicks a cell; background clicks are inert. (test: click (40,40))
- H2: Object cores (13) are interactive; diamonds may be targets/buttons.
- H3: Big frames contain "pictures" (14/11 patterns) that must be matched/completed somehow.

## Probe plan
1. Click background (40,40) -> expect no_impact (sanity check of CLICK semantics).
