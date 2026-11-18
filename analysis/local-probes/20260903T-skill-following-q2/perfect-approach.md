# Perfect approach — unknown ARC-AGI-3 game, this system

The unit is **actor plus host**, not a smarter completion. Intelligence that does not constrain the next `submit_actions` is commentary. The model builds a small machine; the host refuses to play except through that machine.

Contestant prior (`CONTEXT.md`): nothing shipped may encode game-specific object roles, coordinates, visual conventions, goals, resources, or transition rules. The helper is ephemeral per game. The skill stays abstract.

Grounded in q2d on `ls20` (`results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/`) and the NOOA `interactive-game-solver` skill. Counterexamples in `flaws.md`.

---

## The loop. Do not skip a phase because you feel ready.

### 0. Open the right channel

First cell: `grid_array`, colour histogram, bounding boxes. Not the pipe text. Not a 64×64 ASCII novel. One legal action, batch size 1, until you know what moved.

### 1. Find the controllable, the HUD, and the budget

Three questions, answered by one-factor tests:

- What actually translates when I press an arrow / USE / CLICK?
- Which rows change on almost every action even when the rest of the grid does not? That band is HUD. It is not progress.
- What depletes? If something shrinks every action, it is a clock. Do not spend it on walls.

Until those three have at least one failed prediction each, batch size stays 1.

Host: a step whose diff lies only in the HUD band, or is empty, **truncates the rest of the batch**. If that step is GAME_OVER, it is not a no-op.

### 2. `encode` / `predict` is the only state that may plan

Notes are a lab book. They may not move the avatar.

After a handful of `(action, before, after)` triples, write `encode(grid)→z` and `predict(z, a)→z'`. **Retrodict the last step before another action is legal.** Mismatch → rewrite the helper, batch size 1. Match on a short window → you may search.

Never submit a sequence whose predicted `z` path was not computed in the cell.

### 3. Search, then commit

Once `predict` holds, BFS or greedy in latent space for a **subgoal** (a testable `z`, not “explore the right”). `submit_actions` gets that path, rationale literally `predict: …`. After the batch, compare. Divergence at step k kills steps k+1…n.

You do not batch 10 because you are bored. You batch 10 because search returned 10.

### 4. A level-up is a new game until proven otherwise

`levels_completed` changed → the helper is suspect. Spawn, HUD, and object roles may all have flipped. Force a fresh `encode` on the new grid. Transfer only what survives a test. Coordinates do not transfer. Forbidden regions do not transfer until re-observed.

### 5. Death is a constraint, not a vibe

GAME_OVER: write into `z` the last playfield cell and the action that died. Next search treats that `(z, a)` as forbidden. If UNDO exists and is cheaper than a full RESET, use it. Before RESET, record what the attempt confirmed — and that content has to appear in `predict`, or it is discarded.

A row in `hypotheses.md` does not count. `predict(z, LEFT) = DEAD` when `z` is in the trap **does**.

### 6. Carry only mechanics, never a walkthrough

At a level boundary, two lists: what generalizes (a movable stamp, a shrinking bar, a collectible that is a hazard on the next room) and what dies (this room’s columns). The second list does not enter anything that could leak into another game.

---

## Where reasoning belongs

Reasoning earns tokens when it is writing or revising `encode`/`predict`, choosing the subgoal, or explaining a mismatch. It does not earn them narrating a grid numpy already parsed, and it does not replace the host checks.

Thinking-on is a different actor. Measure it on compilation (did search run, did the helper refresh), not on fluency. Do not silently re-enable it inside a host card.

---

## Host refusals — the failures become unrepresentable

| If this happens | The host does |
|---|---|
| HUD-only or empty diff mid-batch | drop the rest of the batch |
| that step is GAME_OVER | do not classify as no-impact |
| `levels_completed` changed | reject submits until `encode` is rewritten against the new grid |
| rationale does not start `predict:` after a trustworthy window | reject the submit (not on turn 0) |
| batch longer than 1 while last retrodiction failed | reject |

Need is discriminating actions under a clock. Form is a latent you can search. Practice is retrodiction every step. Cheap probes (batch 1); keep the helper that predicted; throw away the note that did not.
