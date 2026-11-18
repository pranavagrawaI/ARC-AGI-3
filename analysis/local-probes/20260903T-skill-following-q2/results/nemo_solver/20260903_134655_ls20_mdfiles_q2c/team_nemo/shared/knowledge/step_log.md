
        ## game-3fa32e level 0 (turn16, step8, NOT_FINISHED)
        ### Observed so far
        - Actions: UP/DOWN/LEFT/RIGHT/RESET (no USE/CLICK/UNDO).
        - 8x LEFT applied. Each LEFT = 52-cell diff:
          - color 12 (a 2x5=10px block) shifts LEFT by 5 each step (cols 34-38 -> 29-33 -> 24-28 -> ... -> 19-23).
          - color 3 (big 892px structure) and 9 (45 scattered px) and 11 (bottom band) shift/convert at the edges (world scrolling).
          - colors 0 (3px, ~r31-32 c21-22), 1 (2px ~r32-33 c20-21), 8 (12px r61-62 c56-63), 4,5 DO NOT move.
        ### Hypothesis (untested)
        - Arrows scroll the world horizontally (side-scroller); 0/1 are the fixed-screen player/reticle; 8 = goal/collectible at bottom-right.
        - TEST: press UP -> if vertical objects (9,3,11,12) shift down & 0/1 stay, it's vertical scroll; if 0/1 move up, it's player movement.
        ### Carry-forward
        - World colors 3/9/11/12 are scrollable geometry; 0/1/4/5/8 are screen-fixed. Re-center with RESET if needed.

