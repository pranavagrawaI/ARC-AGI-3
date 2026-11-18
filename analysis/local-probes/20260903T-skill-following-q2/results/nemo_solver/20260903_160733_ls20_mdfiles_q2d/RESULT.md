# q2d result — 20260903_160733_ls20_mdfiles_q2d

Written after the run. The card is `EXPERIMENT-CARD.md` and `PRECOMMIT.md` Addendum 4.
`score_q2.py` was not edited after seeing these numbers.

## Invalidity

Not invalid. 24 agent turns (floor 15). `reasoning_tokens = 0` on all 129 calls.
No `EmptyContentError`. Containment `forbidden_hits: []`. Spend $0.86 of $6.
`20260903_160506_*` remains VOID (PR_SET_PDEATHSIG). This dir is the live arm.

## Gate, on the unit the precommit named

PRECOMMIT: compute rate = turns with ≥1 `repl_execute` before `submit_actions` ÷ turns;
turn contract = turns ending in exactly one `submit_actions` ÷ turns.
`events.jsonl` has an `agent_turn` event. `result.json` reports `turns: 24`.
Those are the same 24 submissions.

| clause | value | threshold | |
|---|---:|---:|---|
| compute_rate | 24/24 = 1.000 | ≥ 0.50 | pass |
| encode and predict helper | `helpers/world_model.py` (`def encode`, `def predict`, `Z_SCHEMA`) | present | pass |
| turn_contract | 24/24 = 1.000 | ≥ 0.90 | pass |

**PASS.** This executor ran this procedure on this game.

`score_q2.py` prints FAIL because it divides by `env_step` count (124), a change
the scorer comments made after seeing q2c force-advances. That unit is not the
precommit's "turns". Against env steps, compute_rate is 0.202 and turn_contract
0.194 — every batched action after the first in a turn has no new `repl_execute`.
The skill itself tells the actor to batch once the model is trusted. Using env
steps as the denominator makes the gate unpassable for a following actor.
Leave the script as it ran; do not retune it to this result.

## Recorded, not the gate

| | |
|---|---|
| retrodiction_rate (rationale starts `predict:`) | 0/24 |
| knowledge | `hypotheses.md` 4473 B, `step_log.md` 6079 B |
| grid_array in emitted cells | 88/126 |
| encode( / predict( / retrodict / load_helpers / write_helper | 6 / 4 / 2 / 6 / 2 |
| helper first written | emitted cell 37, agent turn 6 |
| batch sizes | 1,1,1,1,1,1,2,6,2,4,4,7,10,10,9,10,1,9,1,9,6,8,9,10 (sum 123) |
| levels_completed | 1 (level 0 RHAE 0.840; 24 actions vs baseline 22) |
| game RHAE | 0.030 |
| game_overs | 1 |
| wall | 1492 s |
| termination | max_env_steps (120 cap; 123 steps recorded) |

The helper is a real latent (stack top-left, 11-bar count/left, STEP=5), not a stub.
Rationales never start `predict:`; retrodiction ran in two cells and did not become
the turn-contract string. Gameplay is not the gate. Completing ls20 level 0 with
this model is the first non-zero level this line of work has on Qwen.

## Tempo (standing metric, not the gate)

| | q2c (killed) | q2d |
|---|---:|---:|
| output tokens | 267,057 | 96,829 |
| reasoning tokens | 243,114 | 0 |
| input tokens | 2,331,260 | 7,018,327 |
| committed env actions | 10 | 123 |
| output tok / env action | ~26,700 | 787 |
| output tok / agent turn | n/a | 4,035 |
| OpenRouter spend | (in the $0.86 line of work) | $0.86 |

Son Pham & Mark Barney's state-graph variant died at 644 tokens/action.
787 is still above that line. The thinking-off treatment cut output per action
~34× versus q2c; input grew because the run actually lasted. Cost is now
context, not reasoning.

## Result map

PASS → skills are a live lever for this executor. A B2/WikiSkill card is now
*writable*, still under hold-out public games and the contestant prior (shipped
skill must not encode game-specific object roles). Do not write it in this
file. The helper that passed the gate is ls20-specific; that is allowed in a
run-local `world_model.py` and forbidden in a shipped `SKILL.md`.

## Next, cheapest first

1. No-impact detection as a pure function over consecutive frames, scored offline
   against this trajectory (123 steps, one GAME_OVER, HUD 11-bar erosion). Zero
   tokens.
2. The +8 pipe-column offset still sits in the text channel. This actor mostly
   used `grid_array` (88/126 cells), so the offset may no longer be the binding
   constraint. Confirm against this trace before spending a run on it.
3. WikiSkill handwritten / evolved / sham on Tycho's five holdouts, only after
   (1) and only with a card. Executor is no longer the blocker.
4. Do not train a latent bridge. Do not re-push V16.
