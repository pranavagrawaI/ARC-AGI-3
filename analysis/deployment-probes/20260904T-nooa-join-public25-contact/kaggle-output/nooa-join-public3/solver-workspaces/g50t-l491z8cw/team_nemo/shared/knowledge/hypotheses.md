
# Hypotheses

## H1: Player walks on 0-cells, blocked by 5 and 8
- status: untested
- test: move DOWN from (2,6), check if player moves to (3,6)
- predicting: player moves 1 cell per action, blocked by walls

## H2: Goal is to reach the color-9 region (rows 49-55, cols 43-49)
- status: untested
- evidence: it's the most distinctive non-HUD color-9 structure in the playfield

## H3: The playfield is a maze to navigate
- status: untested
- evidence: walls (5) and obstacles (8) form corridors

## H4: SPACEBAR has a special function (interact/break wall/jump)
- status: untested
- evidence: SPACEBAR is in available_actions but not standard movement

