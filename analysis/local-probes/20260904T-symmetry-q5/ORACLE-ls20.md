# ORACLE LABEL — ls20 mechanics. Quarantined.

**Provenance: Pranav, 2026-09-04, from privileged knowledge of the game.**
Per `CONTEXT.md`, this is an **Oracle Label**: it may evaluate or debug an admissible
contestant. It is **not** training input, and nothing in it may enter a shipped
`SKILL.md`, a prompt, a brief, a seeded helper, or the host.

Handling rule for q5: this file scores the run. It never reaches the actor. Any gate
whose threshold was chosen because of what is written here must say so in the card.

## The mechanic

Move to and interact with **changers**. A changer alters what your **key** looks like.
When the key matches a **lock**, move to the lock to unlock it. Unlock every door to win.

The goal is therefore an **equivalence statement**, not a destination. "Reach the enclosure"
is a wrong theory of this game — it is the theory q3c transferred and defended.

Pranav's generalisation, recorded as a design claim to be tested, not an established fact:
*many ARC-AGI-3 games are "act until something becomes the same as something, fits inside
something else, matches a level".*

## Corroboration from public frames (q3c, 202 steps, my own analysis)

Everything below is recomputed from `steps/step_*.json` and is publicly visible; the oracle
statement told me where to look, not what is there.

| element | location | behaviour |
|---|---|---|
| key glyph | rows 54–60, cols 0–11 — a figure of colour 9 on a field of 5 | changes discretely at steps **19, 69, 76, 130, 132, 137, 178, 184** |
| clock / budget bar | rows 61–62, cols 13–54, colour 11 eroding to 3 | changes on **97.5%** of steps |
| door slots | rows 61–62, cols 55–63, pattern `588 588 588` | flips at steps **76, 95, 137, 141, 184** |

The key is a small figure of one colour on a background. `canon(fig, translate=True)` is
exactly the predicate that decides whether it matches a lock — and it is a **terminal-class**
predicate, the kind q4 requires of a discriminator and q3c never produced (every level-1
`expect` was `move`).

## Why q3c could be 19/19 on dynamics and still wrong about the goal

Its helper (`team_nemo/shared/helpers/game_e5a3ff.py`) declares:

```python
GOAL   = [(r, c) for r in range(45, 50) for c in range(34, 39)]   # hardcoded L0 rectangle
ITEMS0 = [(31, 21), (32, 20), ...]                                # hardcoded L0 constants
def encode(grid_rows): -> {"grid", "mover", "mover_size", "items"}
def in_goal(mover): -> overlaps GOAL
```

**Neither the key nor the door slots appear in `Z_SCHEMA`.** The latent state was
structurally incapable of expressing the win condition, so the dynamics could be perfect
and the objective could not be. `in_goal` tests overlap with a fixed rectangle carried
from level 0.

This is the mechanism behind q3c's headline contradiction, and it is stronger evidence for
q4's premise than the q4 card had available when it was frozen.
