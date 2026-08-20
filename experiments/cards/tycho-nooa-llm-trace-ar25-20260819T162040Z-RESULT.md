# tycho-nooa-llm-trace-ar25-20260819T162040Z — RESULT

Status: VALID. Ran to its stated cap and answered its stated question.

Result: 20 actions, 0 levels, stop_reason `action_budget`, $1.887, 236 LLM calls.
Artifacts: `runs/tycho-nooa-llm-trace-ar25-20260819T162040Z/` (untracked).

## The question it was launched to answer

Does the actor's context survive a 20-action episode with `grid_keep=3`, and does
reasoning reach the model?

Both yes.

| | actor | builder |
|---|---|---|
| calls | 71 | 165 |
| cost | $0.764 | $1.122 |
| peak prompt | 69,580 (53% of window) | 64,767 (49%) |
| shape | monotonic, never reset | 6 fires, each fresh |

Transmission is healthy: 230/236 calls carried tool calls (98%). The six that did
not are builders closing a pass with prose, which is their normal termination. The
comparison is the pre-fix run, 3/26 (12%).

`content` was empty on 230 of 236 calls, and 55% of all generated tokens (57,949 of
104,555) were reasoning. This provider puts effectively everything on the reasoning
channel, so `retain_assistant_turn`'s guard is load-bearing on essentially every
call, not in an edge case.

Prompt cache did the economic work: 5,884,233 of 7,480,081 prompt tokens served
from cache (79%). Without it this run would have cost several times $1.887.

## Context growth: measured, and one projection retracted

The actor's growth is two-term, and the terms are far apart:

| turn boundary | grids held before → after | actor delta |
|---|---|---|
| 1 | 1 → 2 | +9,669 |
| 2 | 2 → 3 | +9,977 |
| 3 | 3 → 3 (oldest evicted) | +1,031 |

One grid observation costs ~9,700 tokens. `grid_keep=3` (`strategy.py:51`) saturates
that term at the third boundary, after which the actor grows ~1,300–1,900 per
action. Final 69,580 of 131,072.

Without the eviction the actor would have crossed the window at roughly turn 13,
inside a 20-action budget. The fix was necessary and is now measured, not assumed.

RETRACTED: mid-run I projected the actor would hit the window near action 8. That
came from two points that happened to straddle the pre-saturation ramp, read as a
rate. It was wrong by a wide margin. Two points are not a trajectory.

Consequence for the deferred proactive event bound: NOOA's reactive archive never
armed because it was never needed. At this budget a proactive bound is not
load-bearing. That is a statement about 20 actions, not about 64 — see below.

## The finding the run was not launched for

The world model is exact, and it scored nothing.

`world_model_report.md` claims `simulation_accuracy=1.00`, 9/9 transitions matched.
The raw transcript corroborates it independently: every directional action changed
exactly 109 cells, and 109 is what the model predicts — two 9x9 components shifting
3 rows change 2 x (9 x 3 x 2) = 108 cells, plus 1 HUD pixel.

It also found something non-obvious without being told: column 63 is a 64-slot move
counter that repaints 11 to 5, one pixel per move, and it correctly worked out that
SPACEBAR increments it while CLICK does not. Colour counts confirm this exactly —
colour 5 went 103 to 121 (+18), colour 11 went 109 to 91 (-18), the same 18 pixels.

And `outcome_verified: no-terminal-yet`. Three goal hypotheses are listed under Open
Hypotheses; none was tested. This reproduces the standing finding (physics modelled,
goals not) with the physics claim now at its strongest: the failure is not a bad
model, it is a correct model that the loop never converts into a goal test.

Not a loop: 19 distinct states in 20 actions, no state revisited. The agent is
wandering, not stuck.

## The budget was binding, and the game said so

The HUD reads 18 of 64 moves used. The game advertises a 64-move budget in its own
grid; our cap stopped it at 18 of them. Those actions are not free — level weights
are 1-indexed and level 0 has weight 1 — but the cap was ours, not the game's, and
it is far below what the game itself permits.

The components ended at rows 39-47, moving down 3 rows per DOWN. The vertical wall's
only opening is at row 63. That is 16 rows, about six more DOWNs. The agent's own
Next Probes section says "perform repeated UP/DOWN steps to find vertical limits";
it was running the right experiment and was cut off partway through.

This does not mean it would have solved the level. It means 0 levels at 20 actions
is not evidence about ar25's solvability, and must not be pooled with the 60-action
positive control as if it were the same measurement. The precommit card said this
before the run.

## The builder goes silent exactly when it stops being wrong

The trigger is `TriggerDispatcher.should_fire -> wm_signal`: it fires when a
committed action's outcome diverges from `world_model.py`'s prediction. There is
no cap and no budget (`trigger_builder.py:45`).

| turns | trigger checks | fired | cause |
|---|---|---|---|
| 1-10 | 10 | 6 | `dynamics` (6/6) |
| 11-19 | 9 | **0** | — |

Once `simulation_accuracy` reached 1.00 nothing diverged, so nothing fired. **Half
the episode — ten actions — ran with zero builder passes.** The agent's whole
reflective capacity switches off at the moment it has a perfect physics model and
no goal.

This is a second regime of the known defect, not a repeat of it:

- model imperfect (the 60-action run): builder fires 30 times, spends the budget
  re-deriving dynamics;
- model perfect (this run): builder never fires again, and nothing asks anything.

Neither regime asks what the goal is. Not because the trigger lacks a goal channel —
it has one — but because that channel is vacuous on this trace. See below.

### Why the goal channel cannot fire

`wm_signal` checks `outcome()` on two sub-branches:

- **too_loose** — `outcome()` claims a terminal on a frame that was not one. Needs no
  observed terminal to fire.
- **too_tight** — `outcome()` misses a real terminal. Gated on
  `level_complete_observable` / `game_over_observable`, i.e. needs one.

The builder wrote:

```python
def outcome(state: State) -> str:
    return "ongoing"
```

A constant. On a trace with zero observed terminals it claims nothing, so too_loose
cannot fire; nothing terminal was observed, so too_tight is unreachable. The probe
prints `OUTCOME:ok`.

**`return "ongoing"` is unfalsifiable against a terminal-free trace — and is therefore
the globally optimal way to silence the supervisor.** Not by intent; by gradient. The
loop's only mechanical pressure is "make the probe say ok", and that function does.

The deadlock: you need to observe a terminal to get the signal that would help you find
a terminal.

The builder was not deluded about this. `notes/world_model_report.md` line 4:

> `outcome: ongoing with unknown terminal condition; hypotheses include aligning
> 4-block over zeros, wall contact, or HUD fill | hypotheses still open`
> `outcome_verified: no-terminal-yet`

It wrote the uncertainty down. But that prose has no mechanical consumer —
`world_model_report.md` appears once in the whole solver, inside
`_tests_constructed_result` (uptake instrumentation). The trigger executes
`world_model.py` and never reads the report. **Honest uncertainty went to a channel
with no reader; false certainty went to the channel the supervisor reads.**

Downstream the same constant kills planning: `subgoals()` returns
`[lambda s: outcome(s) == "level_complete"]` — unsatisfiable — and `heuristic()`
returns 0, so `plan_subgoals` expands to `max_nodes` and returns nothing.

Uptake, over the 6 passes that did happen: the actor followed the recommended
action 3/6 and **tested the builder's report 0/6** (`test_evidence` empty every
time). The builder recommended UP four times; the actor answered DOWN, UP, CLICK,
CLICK.

## The actor inverted the death clock into a progress bar

Goal-uncertainty language in the actor's reasoning peaks precisely in the
builder-silent stretch (turn 12: 8 hits, turn 19: 8 hits). At turn 15 it writes
"I'm stuck." By turn 19 it has settled on a conclusion:

> "maybe the level is simply to fill HUD bar. Let's assume goal is to fill column
> 63 completely (63 moves). We have done 19 moves. Need 44 more. Could do SPACEBAR
> repeatedly."

The builder had correctly identified column 63 as a *move counter*. This is the
`_step_counter_ui` recorded in [[arc-agi-3-attempt-and-action-rules]] — a per-level
step budget that raises GAME_OVER when exhausted. The actor entertained filling it
as the win condition, costed the plan at 44 more SPACEBARs, and then chose DOWN
instead. It did not execute the plan.

What matters is not the plan but that the hypothesis stayed live and untested to the
last action, and that it is exactly inverted: the death clock read as a progress bar.
The one genuinely non-obvious thing the builder discovered was handed to the actor
without its sign, and nothing in the loop could supply the sign or contradict the
guess.

## Reading

Two independent results, and they do not support each other:

1. The context work is done for this budget. Grids dominated, eviction bounds them,
   the actor finishes at 53%.
2. Context was never what was stopping this agent. A model with 1.00 simulation
   accuracy and a discovered move counter still produced no level.

The next question is not "how much context" but "what makes the loop test a goal
hypothesis it has already written down". Any further context work at this budget is
optimising a constraint that is not binding.

Open, and deliberately not started here: how much reasoning to keep (upstream bounds
it at TYCHO_REASONING_KEEP=3), and whether a 64-action budget changes conclusion 2.
