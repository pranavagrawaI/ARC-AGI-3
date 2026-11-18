# Hypotheses (level 0, turn 0)

## Scene facts (confirmed by pixel scan)
- Background 5. Blue 4 = L-frame top-left (col16 rows0-17 + rows16-17 cols0-16 + row 15? see below) + bottom row 63 all blue.
- Red 3 = field rows 0-8, cols 18-63 (359 px), frames the "B" object.
- B: white 15 3x3 at rows 3-5, cols 34-36; border: blue 4 at row2 cols34-36 & row6 cols34-36 & col33 rows2-6 & col38 rows2-6; black 0 at col34? Actually border cells row2 col34-36=4, row6 col34-36=4, col33 rows2-6=3? NO: col33 rows2-6 = [3,4,4,4,3]... need precise.
- A: white 15 5x10 at rows 8-12, cols 3-12.
- Green 2 box: rows 24-32, cols 25-38 border; interior rows 25-31 cols 26-37 = white 15 (84 px).
- Black 0 solid 10x10 at rows 34-43, cols 27-36.
- White total 143 = 50 (A) + 9 (B) + 84 (green interior).
- No visible player token. Candidate actors: A (white block) or the game is click/puzzle-based.

## H1 (untested): arrow keys move the white block A
- Deciding test: submit LEFT; if A shifts 1 col left -> confirmed; if nothing -> refuted for A.
## H2 (untested): CLICK interacts with objects (e.g., click black square fills it / changes color)
## H3 (untested): goal = make some object land in/replace the black square (or green box) — the 10x10 black square matches A? No, A is 5x10 not 10x10. Green interior is 7x12. Black is 10x10; B is 3x3. Nothing matches 10x10 except... A is 5x10. Hmm. Maybe black square is a hole to drop something into.
## H4 (untested): SPACEBAR = interact/use.

## Precise border of B (verified turn 0)
rows 2-6, cols 33-38:
r2: 3 3 4 4 4 4 3
r3: 3 3 4 0 0 0 3   <- wait, printed earlier as "334000": col33=3 col34=4? let me re-verify next turn.
