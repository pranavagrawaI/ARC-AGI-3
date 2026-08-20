# tycho-nooa-no-wm-ar25-20260819T173935Z — RESULT

Status: **VOID — external quota, not a result.** The provider key hit its daily
limit at action 5 of 60. `stop_reason: exception`,
`litellm.APIError: Key limit exceeded (daily limit)`.

None of the card's five measures can be reported. The card predicted measure 1
by turn 3 and measures 2/3/4 as the actual question; a 5-action sample cannot
distinguish "absent" from "not yet". Do not read the nulls below as nulls.

Artifacts: `runs/tycho-nooa-no-wm-ar25-20260819T173935Z/` (untracked). 5 actions,
40 LLM calls, $0.439.

## What the run does establish: the arm is real

This much is not affected by the truncation, because it is a fact about the
apparatus rather than about the agent.

- Workspace contains `wmlib.py`, `notes/`, `level_0/` and **nothing else**. No
  `world_model.py`, `verify.py`, `plan.py`, `predict.py`. The card's "Invalid if"
  condition on seeding is clear.
- `llm-trace.jsonl`: **40/40 calls `source=actor`, zero builder.** The
  `TYCHO_MODE=no_world_model` switch crossed the arc-host env allowlist, reached
  `mode_from_env()`, and propagated to all four call sites.
- Reasoning text present on every call; `TYCHO_NOOA_CAPTURE_REASONING` holds
  under the baseline prompt too.

So the topology probe is wired and can be re-run as-is. Only the quota failed.

## One observation worth keeping, at the strength of five actions

Tool mix over 5 actions: `run_python` 20, `edit_file` 7, `take_action` 5, `ls` 3,
`read_file` 3, `write_file` 2. **Four analysis calls per committed action.**
Removing the builder did not remove the modelling; the actor picked it up in its
own turn. If that survives a full run it reframes the card's result map, because
"apparatus displaces task reasoning" would then be untestable by subtraction
alone — the apparatus regrows.

Two content notes, offered as leads and not findings:

- The baseline actor reached the mirror property by turn 5 unaided: *"LEFT/RIGHT
  move halves horizontally in opposite directions... AAA barrier is static and
  acts as mirror axis."*
- It read the column-63 counter as a **deadline**: *"b barrier erodes from top at
  1 cell per turn... aligning ... before erosion completes."* That is the correct
  sign. The `162040Z` trigger actor, handed the same object as a builder finding,
  inverted it into a progress bar and planned to spend its budget filling it. A
  same-object, opposite-sign contrast is exactly the kind of thing a full run
  should be checked for — at n=1 and 5 actions it is a coincidence until repeated.
- Still absent at truncation: any discriminating probe, any checkable success
  criterion, any mention of Shape B as the body that belongs on the target. The
  standing hypothesis is *"aligning left maze holes with right 4 block"*. This is
  the same failure mode; it is simply too early to say so.

## Cost, and why the card's secondary measure is unanswered

$0.439 over 5 actions is ~$0.088/action, against `162040Z`'s $0.094/action
including its builder. The premise that dropping the builder makes the arm cheap
is **not supported so far** — the actor's own `run_python` habit buys the spend
back. But five actions is on the pre-saturation ramp of exactly the kind that
produced the retracted context projection last run, so this number is recorded,
not believed.

## Next

Re-run the same command unchanged once quota resets. The card stands as written;
nothing about it needs revision, and no code change is indicated. The only open
design question the truncation raises is whether the tool-mix observation above
deserves its own measure in the re-run — a count of analysis calls per committed
action, actor-side, comparable across arms.
