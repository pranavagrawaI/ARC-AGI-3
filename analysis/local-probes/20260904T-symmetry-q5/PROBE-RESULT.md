# Offline separator probe — NEGATIVE. q5 on ls20 is blocked.

Run 2026-09-04 against `../20260904T-host-control-loop-q3/results/nemo_solver/20260904_140403_ls20_mdfiles_q3c/`.
Zero tokens. This is the test PLAN §3 specified and that I had skipped after the oracle
disclosure; the oracle may choose our game, it may not stand in for this.

## What was tested

Predicate: the HUD key figure (colour 9 in rows 54–60, cols 0–11) is congruent to some
connected component in the playfield (rows 0–53).

- all 16 colours as the component mask
- `canon(..., point='none', translate=True)` — position-free congruence
- `similar(...)` — congruence up to k-fold magnification, since the HUD key is drawn
  at 1× and playfield objects are 5× blocks
- against every one of the 203 recorded frames, and separately against
  `ipc/level_0_terminal.json`'s recorded `pre_win_grid_rows` and `terminal_grid_rows`

## Result

**No match anywhere.** 0 of 203 frames. 0 at the pre-win frame. 0 at the terminal frame.

At the terminal the key is 24 cells, primitive core 6 at k=2. The playfield holds
components of size 5, 15 (colour 9), 10 (colour 12), 8, 8 (colour 11), 43, 208 (colour 5)
— no congruence, no similarity, at any colour.

An earlier pass tested `g[32]` as the terminal. That was wrong: the level changes **at**
step 32, so `g[32]` is already level 1. Redone against the recorded terminal frames; the
answer did not change.

## What this licenses

It does **not** falsify the oracle mechanic — Pranav knows the game.

It does mean the specific recoverable predicate q5's S2 depends on could not be
constructed from public frames in a bounded attempt. Under Mayo's criterion this
hypothesis was put to a test it would have passed if true, and it failed. That is
evidence against *this* predicate, not proof that no separator exists.

Live alternatives, none tested: the key region may include decorative frame cells and not
be the key; level 0 may predate the changer mechanic entirely (it completed on `UP` after
32 actions, and q3c's own note read *"Items (0/1 cross) = KEY to unlock box entry"*); the
lock may not be on-screen at level 0; the rendering scales may not be integer multiples.

## Consequence, per the rule set before the probe

PLAN §3: *"If no relation separates them, q5 on ls20 is the wrong game."* That rule stands
and it binds now. **Do not freeze the q5 card against ls20.**

The next step is more offline probe, not a run — it costs nothing, and the budget makes
the point moot anyway (remaining $2.75 against a $6 floor). Cheapest first:

1. Re-derive the key region from change structure rather than my hand-drawn box, and
   re-test.
2. Test on a **level-1** frame pair — the changer mechanic is what the oracle describes,
   and level 0 may not exercise it. q3c never completed level 1, so there is no recorded
   terminal; use the 8 key-change steps as the anchor instead.
3. If a separator is found, q5 proceeds with it as the scoring key. If not, q5 moves to a
   game where congruence demonstrably decides, and `ar25` is the one with prior evidence.

---

## Holes closed 2026-09-04 — still negative

Script: `probe_holes.py`. Same 203 frames plus recorded `pre_win` / `terminal`.
Key = colour-9 cells in rows 54–60, cols 0–11. Door band = rows 61–62, cols 55–63
(three 2×3 slots).

### A. `colours="any"` vs playfield CCs

The first probe used the default `colours="fixed"`. A same-shape lock in another
colour would have been invisible.

**0 hits** for `canon(point='none', translate=True, colours='any')`.
**0 hits** for `similar(..., colours='any')` (D4 + scale).
Terminal playfield sizes unchanged: (4,1807), (3,1334), (5,208), (5,43), (9,15), …

### B. Key vs door slots

Door band on L0 is `588588588` / `588588588` — three filled 2×3 slots. First
flip is step 76 (L1): slot 2 becomes `533`/`533`. Key is 24 cells, primitive
core 6 at k=2:

```
key          core
999999       999
999999       9..   (after step 19: ..9)
99....       9.9
99....
99..99
99..99
```

Compared key and core to whole band, each slot, and door-band CCs, under
fixed/any × none-congruence/similar.

**0 hits.** Structural: a 2×3 filled rectangle is not the 3×3-with-holes core,
even after recolouring. `similar` cannot make them the same.

Door-slot figures vs playfield CCs with `colours='any'`: **0 / 0**.

### What this does not close

The key HUD does change on L0 (step 19, 8 cells). Doors do not flip until L1
step 76. So L0 is not a no-key-animation prefix; it is a no-door-flip prefix.

Still untested: re-derive the key from the eight co-changing cells rather than
the hand-drawn box (the box is now corroborated — those eight steps are exactly
the cells in this region), and test whether the *key's own successive forms*
are congruent to anything at the changer steps on L1. That is a different
predicate (key_t ≡ playfield_object, or key_t ≡ key_{t-1} after a changer).
q5 on ls20 stays blocked until one of those separates.

---

## Eight key-change steps — 2026-09-04

Script: `probe_l1_keychange.py`. Change steps re-derived from the key box:
**19 (L0), 69, 76, 130, 132, 137, 178, 184 (L1).** Matches the oracle list.

### What matched

Successive HUD keys are **D4-congruent** at all eight steps
(`congruent` / `similar`, `colours` fixed or any). `canon(point='none')` does
not hit: the glyph **turns**, it does not stay rigid. Cores:

```
999      999      9.9      9.9
9..  →   ..9  →   ..9  →   9..
9.9      9.9      999      999
```

That is the key's own orbit, not a lock match. It is also true of identical
successive keys (identity ∈ D4), so it does **not** separate terminals.

### What did not match

At every change step, against playfield CCs at t and t−1, door band, and each
2×3 slot, under none/congruent/similar × fixed/any:

**key_t vs play = 0. core_t vs play = 0. key_t vs door = 0. core_t vs door = 0.**
Changed-playfield-cells as a figure vs key/core: 0.

### What else moves with the key

Every change step has ~50 playfield diffs that are a **5-wide sprite**
(`ccccc` / `99999` / `33333` bands) — the player tile overlapping a 0/1-cross
object, not a 6-cell key core. Door flips co-occur at 76, 137, 184 (slot 2
`588`→`533`), never on L0. Step 19 is the L0 item pickup from q3c's own note.

### Consequence

The L1 changer steps do not give S2 a recoverable key≡lock predicate on these
frames. Symmetry describes the HUD glyph's pose. It does not identify a lock
in the playfield or the door band. q5 on ls20 stays blocked. Next game, if
any, is one where a separator has been written down first (`ar25` has prior
uptake). Do not freeze a card.

---

## L1 start pair — 2026-09-04, after a location hint

4-connected (and even 8-connected) CCs **split** the playfield 9-glyph
because its hole is a full cell of 5. Taking all colour-9 cells in the
5-box at rows 39–45, cols 13–20 yields one 6-cell figure. That is the
extraction the first probe did not do.

L1 step 32:

```
HUD key (k=2, n=24)     playfield 9s (k=1, n=6)     top 11-ring (n=8)
999999                  999                         bbb
999999                  9..                         b.b
....99                  9.9                         bbb
....99
99..99
99..99
```

`similar(HUD, playfield 9s, colours=any)` is **True**. Cores are D4-congruent
(reflection). `point='none'` is false — size and rotation both differ, as
stated. Holds on later L1 frames while the 9-glyph stays 6 cells; breaks
when the window picks up extra 9s (player overlap, steps 50 and 202).

The two 11-rings (top at (16,15) and lower at (51,40)) are **copies**, same
pose, `none` and `fixed` both true. They are not similar to the 9-glyph
(8 vs 6). If the intended pair was 9-glyph vs 11-ring, that pairing is
false. If it was HUD vs boxed 9s, it is true — and 4-connected search hid it.
