# No-impact offline + offset check, on q2d

Trajectory: `../results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/`
Card: `EXPERIMENT-CARD.md`. Scorer: `score.py`. 0 API calls.

Room names below follow the SDK: first room = level 0 (completed), second room = level 1 (stuck). If you are counting rooms 1 and 2, those are the same two rooms.

## The three claims, against the frames

**1. Figured-out knowledge stayed in notes, not in machinery.** Holds.

- `hypotheses.md` H29: wide bar cols 14–18 is a death trap, *confirmed*. Turn 21 then submitted LEFT×8 “to align with left arm (cols 34→14)”. Step 103 is GAME_OVER, stack at cols 9–13.
- `world_model.py` `encode`/`predict` written at turn 6. 4 `predict(` and 6 `encode(` in 126 emitted cells. 0 of 24 rationales start `predict:`.
- `predict` on RESET still returns `(45, 34)` and `bar_cells=84` — the first-room spawn. After the level-up the spawn is `(40, 29)` with 100 bar cells (step 24 and again after the death reset, step 104). The helper was not rewritten.

**2. Crucial information was not used as a signal.** Holds, and it is specific: *blocked and timer-only steps were eaten as if they were moves.*

Turn 9 submitted UP×3 into the shaft. Steps 18–20 changed 2 pixels each, both in rows 61–62, colour 11→3. The block did not move. The batch continued. Same pattern on the second room: 31 HUD-only steps inside 10-action batches.

The 11-bar as a 42-move budget was written down (H9, H34) and then a 10-action batch ran it to empty (turn 18).

**3. No recovery; the second room is not an accidental win.** Holds on this trace, not as a theorem about ls20.

After GAME_OVER the harness RESET. Step 104 stack is again `(40, 29)` — the second-room spawn, not the first. Turn 22’s rationale is still “UP to the wide bar, DOWN to the middle, avoid cols 14–18”. The helper still predicts the first-room RESET. 99 second-room steps, 0 completions. 32 of those 99 are blocked or HUD-only.

## Item 1 — no-impact predicate

HUD rows discovered by change frequency ≥ 0.8: **61, 62** (121/123 pairs). Not hardcoded.

| label | n | what the pixels did |
|---|---:|---|
| playfield | 88 | stack (and usually the bar too) |
| hud_only | 34 | only rows 61–62 |
| blocked | 1 | step 41, zero cells |

Every step with ≤4 changed cells is `hud_only` or `blocked`. No tiny step is labelled playfield. No HUD-only step has n>4 except none — wait, all 34 HUD-only have n∈{2,4}.

**Exception, predicted:** step 103 is `hud_only` (4 cells, rows 61–62, colour 8→3) **and** GAME_OVER. HUD-only is not a safe no-op. A host that retries HUD-only without looking at `state` would hide a death.

Second room is where it matters: 3 HUD-only in 23 first-room steps, 31+1 in 100 second-room steps (32%).

This is one game, one run. It does not estimate Son Pham’s +55%. It does show the predicate is *detectable* here without naming colour 11, and that shipping it requires the terminal-state exception.

## Item 2 — +8 pipe offset

Does not bind on this trace.

Numpy top-left of colour 12 at step 0 is `(45, 34)`. The +8 bug would have written `(45, 42)`. The notes say `initial(45,34)`. The helper RESET uses 34. No `col 42` in the knowledge files. 88/126 cells call `grid_array`.

Do not spend a run on the pipe format until an actor that *reads the pipe* is the one playing.

## Item 3 — WikiSkill

Not written. The diagnosis you just gave does not point at skill text.

A better skill file would still land in markdown. The failures above are: a death-trap note that did not constrain the next batch; a helper that did not refresh on level-up; HUD-only steps that did not stop the batch. Those are host predicates and a re-encode trigger, not an evolved `SKILL.md`.

The Q2 gate made a WikiSkill *card* writable. It did not make it the next experiment.

## Result map, followed

| result | action |
|---|---|
| HUD split clean + death exception | host-side predicate is writable (HUD-only ∨ blocked, abort if GAME_OVER). Not shipped from this file. |
| offset not binding | drop the pipe-format run |
| WikiSkill not implied | do not write that card next |

Next card, if we write one: host rejects or truncates a batch at the first HUD-only/blocked step, and forces `encode` again on `levels_completed` change. Same contestant prior: no game-specific object roles in anything that ships.
