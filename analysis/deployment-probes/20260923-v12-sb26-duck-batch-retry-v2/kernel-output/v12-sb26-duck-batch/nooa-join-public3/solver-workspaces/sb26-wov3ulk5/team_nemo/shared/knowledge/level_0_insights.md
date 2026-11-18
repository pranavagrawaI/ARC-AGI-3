# Level 0 Insights

## Winning Sequence
1. Click each of the 4 bottom pieces to select it (20px change)
2. Click the corresponding slot in the middle frame to place it (53px change)
3. After all 4 slots filled, press SPACEBAR to confirm (755px change, level_up)

## Mechanics Discovered
- **Selection**: Click a piece in the bottom area → it becomes "selected" (20px visual change)
- **Placement**: Click a slot in the middle frame → the selected piece moves there (53px)
- **Confirmation**: SPACEBAR after all slots filled → level completes (755px)
- **Blocked clicks**: Clicking non-piece areas (gray slot markers, empty space) → blocked/no_impact
- **Batching**: 2-action batches (select+place) work reliably

## Color Matching
- 4 bottom pieces: blue(9), green(14), yellow(11), purple(15)
- 4 middle slots: filled in order left-to-right
- 4 HUD windows: black(5), green(14), yellow(11), purple(15) frames
- The pieces match the HUD window colors (except slot 1=blue vs window 1=black)

## Open Questions
- Does the color of each piece need to match a specific slot? Or any slot works?
- What does the SPACEBAR progress bar (row 53) track?
- Are there multiple levels and how do they differ?

## Actions Spent: 16 (including exploratory actions)
