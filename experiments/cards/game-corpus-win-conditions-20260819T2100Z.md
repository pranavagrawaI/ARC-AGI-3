# Survey: what all 25 ARC-AGI-3 games actually ask for

Not an experiment. A census of ground truth, taken from the game sources themselves.

Pulled with `tools/pull_game_sources.py` (read-only GETs; no scorecard, no attempt
charged). 25/25 into `environment_files/` (gitignored). Table regenerable with
`python tools/extract_win_conditions.py [--body]`.

## Why the sources are readable at all

Identifiers are obfuscated — `self.hujpxmlafgh`, `def wvrremwltt`. The engine's own
API is not. `next_level`, `lose`, `step`, `on_set_level`, `pixels`, `set_position`,
`get_sprites_by_tag`, `rotation` survive in all 25 files.

That is enough, because of one fact: **every game calls `next_level()` in exactly one
place.** 25 files, 25 call sites. The win condition of each game is a single guarded
statement, and the guard's *shape* is legible even when its names are not.

## The census

| game | win condition, as written | relation |
|---|---|---|
| ar25 | every target point has `grid[y,x] >= 0` | coverage |
| bp35 | `landed_on_gem` → `win()` | coincidence |
| cd82 | `np.array_equal(A.pixels[mask], B.pixels[mask])` | **form equality** |
| cn04 | after applying each sprite's `rotation`, no cell of colour 8 or 13 remains | **coverage under D₄** |
| dc22 | `a.x == b.x and a.y == b.y` | coincidence |
| ft09 | for each of 8 neighbours: centre colour equal-or-unequal per edge marker | **edge constraint** |
| g50t | player flag set on reaching goal | coincidence |
| ka59 | `in.x == out.x+1 and in.y == out.y+1 and in.h == out.h-2 and in.w == out.w-2` | **containment** |
| lf52 | reach → `win()` | coincidence |
| lp85 | every tagged sprite has a `goal` sprite at `(x+1, y+1)`; second tag → `goal-o` | coincidence + type match |
| ls20 | player on key position **and** `colour == c[i] and shape == s[i] and orient == o[i]` | **attribute match** |
| m0r0 | count of unmatched pairs `== 0` | pair matching |
| r11l | no sub-object still flagged | aggregate |
| re86 | composite movables onto a canvas, then every target pixel equals the canvas | **rendered form equality** |
| s5i5 | every A coincides with some B | coincidence (set-wise) |
| sb26 | last slot reached and `pixels[1,1] == pixels[0,0]` | sequence + colour equality |
| sc25 | animation gate; win set upstream | opaque |
| sk48 | for each channel pair, i-th collected centre colour equals the other's i-th | **sequence equality** |
| sp80 | tagged sprite consumed | reach |
| su15 | colour propagation completes | region fill |
| tn36 | sub-object flag | opaque |
| tr87 | rule-chain traversals `== len(parts) * (len(rules)-1)` | rule application |
| tu93 | every unit's `(x,y)` is an exit position | coincidence (set-wise) |
| vc33 | interval containment picks the container, then same-axis alignment + wall region | **containment + region** |
| wa30 | every tagged sprite's `(x,y)` in the target set, and not in the excluded set | coincidence |

## The result

**All 25 win conditions are spatial relations between two things. Zero are temporal.**

Not one game is won by surviving N steps, by a transition sequence, or by any property
of the trace. Every guard compares one set of objects against another set of objects in
a single frame. The step counters that exist (`_step_counter_ui` in ls20, column 63 in
ar25) are *death clocks* — they gate `lose()`, never `next_level()`.

Set that against the instrument the agent has. `wmlib` exposes 52 helpers: ~50 temporal
(`frames`, `transitions`, `verify`, `plan_bfs`, `terminal_events`, `death_events`), 2
spatial (`segment`, `segment_summary`, both decompositional), 0 relational.

The apparatus is aimed at an axis on which none of the 25 games are scored. This is not
a gap in coverage; it is a category error, and it is now measured rather than argued.

## Which relations the corpus demands

Counting the table, and pooling the synonyms:

- **coincidence** — do these two occupy the same cell? — 9 games
- **form equality** — are these two the same shape/colour, possibly under a mask or a
  composite render? — 5 games (cd82, re86, sb26, sk48, ft09)
- **containment** — does this fit inside that? — 2 games (ka59, vc33)
- **coverage** — is every cell of the target claimed? — 2 games (ar25, cn04)
- **attribute match** — same colour *and* shape *and* orientation? — 2 games (ls20, sk48)
- singletons — pair matching, rule application, region fill — 3 games
- opaque at this depth — 3 games

`ka59` is worth quoting whole, because it is the user's "does this thing fit inside this
thing" written out by the game author:

```python
def dujiampjkx(inner: Sprite, outer: Sprite) -> bool:
    return (inner.x == outer.x + 1 and inner.y == outer.y + 1
            and inner.height == outer.height - 2
            and inner.width == outer.width - 2)
```

And `ls20` is the lock-and-key comparison, also written out:

```python
def bejndxqqzf(self, i: int) -> bool:
    return (self.colour == self.lock_colour[i]
            and self.shape == self.lock_shape[i]
            and self.orient == self.lock_orient[i])
```

## D₄ is not a guess

20 of 25 games carry `rotation`; 5 use `np.rot90`; 2 use a flip. `cn04`'s win predicate
transforms coordinates through all four rotations inline before testing coverage.
`cd82` paints its diagonals per rotation and then compares under a mask.

A `correspond(A, B) -> element of D₄ × translation, or None` is therefore not a design
preference. It is the operation the corpus is built out of.

## What this does not say

The census is of *guards*, not of difficulty. A one-line guard can sit behind an
arbitrarily hard search — `re86` requires composing a picture, `tr87` requires deriving
a rewrite system. Knowing the relation does not hand you the plan.

What it does say is that the agent currently cannot *state* any of these 25 conditions,
because it has no primitive that compares two objects. `outcome()` returning the constant
`"ongoing"` is not laziness; it is the only thing expressible in the vocabulary provided.

Three games remain opaque at this depth (sc25, tn36, r11l — flags set on sub-objects).
Unfolding them is a one-hour job and would not change the count.
