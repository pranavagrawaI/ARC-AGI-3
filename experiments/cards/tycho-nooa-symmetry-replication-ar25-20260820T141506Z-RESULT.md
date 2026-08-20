# tycho-nooa-symmetry-replication-ar25-20260820T141506Z — RESULT

## Judgment

Level-0 completion and symmetry uptake replicated locally on ar25 seed 0, but
not by an identical trajectory. The agent completed on action 20/20 after two
successful `symmetry.canon` reaches. This raises the claim from one anecdote to
a same-configuration replication; it does not establish transfer across seeds
or games.

## Execution

- result: `levels_completed=1`, `actions=20`, `stop_reason=action_budget`
- route: `CLICK(17,40), UP, RIGHT, DOWN x11, LEFT x6`
- model-derived route after the probes: from `(12,33)` to target `(45,51)`,
  hence `DOWN x11`, `LEFT x6`; execution matched exactly
- symmetry: actor reached `canon` twice and both calls completed; builder reached
  it zero times
- model traffic: 215 calls, 7,627,800 prompt tokens, 103,541 completion tokens,
  `$2.392786`; actor 87 calls / `$1.310921`, builder 128 / `$1.081865`

The earlier credited attempt `135428Z` is invalid, not a negative replication:
after three actions, builder prose `recommended_action: CLICK` was parsed into
an invalid coordinate-free `ActionDecision` and crashed before actor control.
Its cost was `$0.647863`.

## What the trace says

The actor used colour-invariant canonicalization to establish that the movable
4-shape and b-target were congruent, discovered the 3-cell movement quotient,
identified that LEFT moves the right half right, and then executed its computed
route. This is legible mechanism evidence rather than success inferred from the
score alone.

The builder fired four times. The first two passes changed the model and the
actor followed 1/2 recommendations. The last two exhausted 43 tool calls each,
mostly `run_python`, forced both report and notes checkpoints, and produced no
model edit; the actor ignored the SPACEBAR recommendation and continued its
already-correct route. Builder work was 59.5% of LLM calls and did not cause the
winning suffix.

## Bugs found and repaired

1. Malformed builder advice can no longer terminate the actor lane. Invalid or
   unavailable recommendations are recorded as parse errors and exposed as no
   recommendation.
2. The protocol reader joins calls and executed tools by `tool_call_id` and now
   renders absent recommendations/errors without crashing.
3. Symmetry uptake requires tool status `complete` and no traceback.
4. Host-confirmed completion now invokes the mandatory outcome checkpoint from
   `action_result`; it no longer depends on a next actor turn. Failures preserve
   the win and remain pending for level-start retry.

## Remaining uncertainty and next action

This is one game and one seed, and the successful retry ended exactly at the
cap. The pre-fix run still left `outcome()` constant because no post-win actor
turn existed; the direct action-result checkpoint repair is test-verified but
has not yet been exercised in a live run. Next run should use a different seed
or game, a larger action cap, and enough budget to observe both transfer and the
post-win checkpoint. Reduce or stop repeated builder passes when they exhaust a
full pass without a model edit.
