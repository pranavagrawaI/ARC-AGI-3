# q6-tool-use-contact result — 20260904_233400

Card: `EXPERIMENT-CARD.md`. Not invalid: digest `7e0c1f1c…`, `reasoning_tokens=0` on all 16 LLM calls, GRID in the interpreter, game names redacted from the sent source.

Spend: $0.048 of $0.12 (credit $0.160 → $0.112). 8 cells × 2 trials, both hit the round cap still calling tools; neither wrote `PAIR:`. The gate is wrapped calls, not that line.

## Gate

| trial | frame | U1 | U2 | what they compared that returned True |
|---|---|---:|---|---|
| A | ar25 L0 turn 0 | 265 | **True** | 45-cell colour-4 vs 45-cell colour-11, `congruent(translate=True, colours='any')`; nonempty `transporter` |
| B | ls20 L1 step 32 | 74 | **False** | two colour-11 8-cell rings, `colours='fixed'`; three colour-8 2×2 blocks. Target 24-vs-6 colour-9: **0 calls** |

Extraction on both trials was 4-connected CCs. The scorer's `cc+window` flag is from `GRID[` in print code, not a window figure.

## A — they found the pair

Round 1: 4-connected components of cells **not** colour 9 (background). Four components, three of them 45 cells: `{0,5}`, `{4}`, `{11}`.

Round 2: `import symmetry` — probe ImportError (the name was already in scope; this cost a round). Recovered in round 3 by using `symmetry` without importing.

Round 3: `colours='fixed'` → all three pairs False (4 ≠ 11).

Round 5: `colours='any'` → colour-4 vs colour-11 **True**. Round 6: transporter, 2 elements. Round 7: same comparison restated.

This is the Tycho uptake pair, including the flattening.

## B — they found the CC-visible copies

Round 0: pasted the hex dump into a string, SyntaxError. Round 1: used `GRID`.

Colour-9 4-connected components on this frame are sizes **5, 15, 1, 20, 4**. The HUD glyph and the playfield 9-glyph both have holes of 5, so CC splits both. There is no 24-cell CC and no 6-cell CC.

They listed those fragments (round 3), printed the two 11-rings (round 4–5), and in rounds 6–7 called the module on:

- the two 11-rings (n=8, n=8) → congruent True, `colours='fixed'`
- the three 8-blocks (n=4) → congruent True
- 9-comp-0 (5 cells) vs 9-comp-4 (4 cells) → False

They never built the colour-9 window at rows 39–45, cols 13–20, and never compared HUD n=24 to playfield n=6. `colours='any'` was not used on B; they did not need it for same-colour copies.

## Probe defects (do not interpret as actor failures)

- `import symmetry` was blocked. They recovered.
- 8-round cap cut both trials off without a `PAIR:` line. A already had U2.

## Card map

**U1 ≥ 1, U2-A = 1, U2-B = 0.** Extraction method is the live failure.

Permitted next: an extractor in the seed, or ar25 as the live game — own card, after credit reset. Do not rerun ls20. Do not spend a game run on `read_helper` in the workflow; this probe already put the source in the cell.

Remaining OpenRouter ≈ $0.11 of today's $10.
