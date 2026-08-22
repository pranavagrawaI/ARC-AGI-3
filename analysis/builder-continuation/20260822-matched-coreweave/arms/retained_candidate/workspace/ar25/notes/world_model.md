# World Model for the "Gap-Filling" Game

## Observed Dynamics
- **Budget:** Column 63 contains a budget bar of color 11. Each movement action (UP, DOWN, LEFT, RIGHT) consumes one unit (11 -> 5).
- **Terrain:** The grid contains "holes" (color 10) in a vertical strip (cols 30-32). These holes are transparent/dominant: if an object moves over them, the hole (10) is still visible.
- **Moving Objects:**
    - **C4 (Color 4):** An L-shaped block.
        - Shape: Top horizontal bar (3x9) and a bottom vertical bar (6x3).
        - Movement:
            - `RIGHT`: Moves left by 3 (x -= 3).
            - `DOWN`: Moves down by 3 (y += 3).
        - Note: In T4 (RIGHT), the bottom part of C4 disappeared, but it reappeared in T5. This suggests a possible state change or a more complex transformation. Wait, looking closer at the frames, in T4 the bottom part of C4 is just missing from the observed slice, but the top part is there.
    - **C5 (Color 5):** An L-shaped block (inverted).
        - Shape: Top horizontal bar (3x9) and a bottom vertical bar (6x3) on the right side.
        - Internal Holes: C5 has a specific pattern of color 0 cells inside it.
        - Movement:
            - `RIGHT`: Moves right by 3 (x += 3).
            - `DOWN`: Moves down by 3 (y += 3).
        - Terrain Interaction: Like C4, C5 is transparent to terrain holes (color 10). Terrain holes (10) also override C5's internal holes (0).

## Current Model State
- `State` tracks `y5, x5, y4, x4` and `budget`.
- `render()` simulates the blocks and budget, preserving terrain holes (10).
- `transition()` currently implements simple translation by 3 units.

## Open Questions / Hypotheses
- **The T4 "Disappearance":** In T4, C4's bottom part disappeared. I need to determine if this is a permanent change, a phase-based thing, or if my `render()` just needs to be more precise.
- **Goal:** The objective is unknown. I need to observe a level completion or a GAME_OVER.
- **C4/C5 interaction:** Do they interact when they overlap?
- **The "Budget" limit:** What happens when the budget reaches 0?
