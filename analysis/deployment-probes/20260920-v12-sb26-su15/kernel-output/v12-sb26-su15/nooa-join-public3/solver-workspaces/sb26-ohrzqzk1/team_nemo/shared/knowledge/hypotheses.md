## H1: Top-tile matching
Top tiles (14,15,11,9) must be matched to bottom blocks (14,15,11,9) in some order.
- Evidence: Same 4 colors appear in both top and bottom regions
- Tested scope: Visual inspection of level 0 initial state
- Open: What action performs the matching? What is the correct order?

## H2: SPACEBAR is a "start" or "submit" action
- Evidence: Available in initial state
- Tested scope: None yet
- Open: May start the game, submit a selection, or have some other effect

## REVISED: Trail is a full-row conversion (step 15)
**Measured facts:**
- Initial (step 0): row 53 = 63 cells '2' (cols 0-62) + 1 cell '3' (col 63)
- After 5 SPACEBARs (step 15): row 53 = 58 cells '2' (cols 0-57) + 6 cells '3' (cols 58-63)
- Each SPACEBAR converts exactly one '2' to '3', boundary moves left
- Row 53 has NO background cells - it's a full 64-cell row

**Hypothesis:** Win condition = convert all '2' to '3' (entire row becomes '3').
- Requires 58 more SPACEBAR presses
- This is the only action that changes the playfield
- All other elements (blocks, tiles, dots) are non-interactive or only affect selection

**Open conditions:**
- What role do the blocks/tiles play? (possibly decorative or for a different level)
- Is RESET a fallback if the conversion goes wrong?
- What happens when the last '2' converts to '3'?

