## Your working state

Keep exactly these eight lines in NOTES.md. Rewrite the whole block every round.
Nothing else you compute survives to the next round; this does.

Replace this whole file with just the eight lines. You do not need to keep these
instructions: the rules below and your last accepted block are repeated to you
every round in `competence.reason_authoring_schema.working_state`.

```
PLAYER_BINDING:
LOCAL_OBJECTIVE:
ROUTE1:
CURRENT_APPROX_POSITION:
LAST_ACTION:
LAST_OUTCOME:
NEXT_STEP:
CONTRADICTION:
```

- `PLAYER_BINDING` — which object you control, and its attributes. `unknown` is
  allowed. Once it is not `unknown` it is settled.
- `LOCAL_OBJECTIVE` — one near-term target. Not the level, not the game.
- `ROUTE1` — the remaining action ids you intend to take, comma-separated.
- `CURRENT_APPROX_POSITION` — approximate is fine. `unknown` is allowed.
- `LAST_ACTION` — the action id you took last.
- `LAST_OUTCOME` — one of `moved`, `blocked`, `special_event`, `uncertain`.
- `NEXT_STEP` — the single action id you are taking now.
- `CONTRADICTION` — empty, or one sentence naming the transition that broke a
  settled binding.

## Order of work

One stage per round. Do not run ahead.

1. Establish one provisional `PLAYER_BINDING`. Provisional means good enough to
   act on, not proven.
2. Build a coarse traversability map — passable or not, nothing finer.
3. Record one `LOCAL_OBJECTIVE` and a provisional `ROUTE1`.
4. Take one action.
5. Classify the consequence into `LAST_OUTCOME`.
6. Continue `ROUTE1`, or repair it locally.
7. Reopen identity or the global reading only after a `CONTRADICTION`.

## Rules the host enforces

A rejected commitment comes back naming the rule. It is not a hint about the
game; it says your new block is not a legal continuation of your old one.

- **R1** A settled `PLAYER_BINDING` changes only with a `CONTRADICTION` in the
  same block, or when `LAST_OUTCOME` is `special_event`.
- **R2** After `moved`, neither `PLAYER_BINDING` nor `LOCAL_OBJECTIVE` may
  change unless you name a `CONTRADICTION`. An ordinary move is not by itself
  evidence about what anything is.
- **R3** After `blocked`, edit `ROUTE1`. A wall is a fact about one cell, not
  about the world — again unless you name a `CONTRADICTION`.
- **R4** `PLAYER_BINDING`, `CURRENT_APPROX_POSITION` and `ROUTE1` may stay
  `unknown`. Unknown is not a contradiction and does not license rebinding.
- **R7** `NEXT_STEP` is exactly one legal action and equals your chosen action.
- **R8** A `CONTRADICTION` licenses one reinterpretation. Clear it next action.
- **R9** At most two consecutive `uncertain` outcomes.

Replacing `ROUTE1` is recorded but never rejected. Re-planning once you learn
the map is ordinary play.

R5 is an instruction, not a host-enforced rule: inspect what changed only after
`special_event`. The submitted state block cannot prove what you inspected, so
the host does not pretend to verify it.

## Spending a round

Reading the frame is cheap; re-deriving what you already settled is not. If
`PLAYER_BINDING` is set, do not recompute it. If `ROUTE1` is non-empty and
`LAST_OUTCOME` is `moved`, take its head and move. Inspect only when the block
says `unknown`, or `LAST_OUTCOME` is `special_event` or `uncertain`.

Finish every round with `return_result`. A round ending in `execute_python`
with no committed action spends budget and changes nothing.
