# What the solver appeared to believe — public endurance v2

## Finding

The solver did not simply stop thinking or stop executing. Its observable
decision process failed differently in each game:

- **LS20:** learned useful movement facts, but repeatedly reconsidered possible
  goals without turning that uncertainty into another action.
- **S5I5:** adopted an unverified progress proxy, then became efficient at
  pursuing it. Predicting a bottom bar replaced discovering the objective.
- **CD82:** narrated movement unsupported by the pixels, while a bug in its
  generated predictor made some unchanged-object observations appear to validate
  that narrative.

The common boundary is not simply lack of computation. It is the failure to keep
**observation, prediction agreement, and objective progress** distinct. This is
a trace-supported interpretation, not proof that one intervention would solve
the games.

## Scope and method

Run: Kaggle version 6, notebook v7, the operationally valid public endurance v2.
There are 153 recorded Actor calls, all with empty response_content, zero reported
reasoning tokens and no reasoning content. What we can analyze is the generated
execute_python code, its comments, action rationales, helper programs, written
notes, and host-recorded observations/prediction checks—not hidden thoughts.

I inspected the records structurally across all three sessions and followed the
pivotal decisions through code/comments, notes, action transitions and checks.
Call numbers below are one-based line numbers in model_calls.jsonl. They are not
game steps: many calls occurred without an intervening environment action.

The deterministic [check_trace.py](check_trace.py) recomputes pixel changes and
replays the inspected, hash-bound CD82 helper on saved public observations.
[FACTS.json](FACTS.json) records the results. No model calls, new game experiments,
environment implementation inspection, or solver changes were made. Full tool
stdout is not present in tool_calls.jsonl, so I do not manufacture a count of
Python errors from the code-repair sequences.

## 1. LS20: real correction, followed by goal speculation

Evidence: [model calls][ls-calls], [transitions][ls-transitions],
[step log][ls-notes], [prediction checks][ls-checks].

**Initial theory.** The 0/1 dot was the avatar, color 12 was a destination, and
movement was one cell through a maze. It spent 13 calls before the first action,
which arrived 91.43 seconds into the session. This includes visual inspection,
helper construction/repair, and attempts to use the experiment interface; not
all of it should be counted as gratuitous deliberation.

**Reality corrected the theory.** UP did not move the dot. It shifted a 5x5
color-12/color-9 structure upward by five cells and changed a bottom bar. The
model examined differences, explicitly marked its old avatar theory contradicted
(call 23), tested DOWN for reversibility, and tested LEFT/RIGHT. It also corrected
an encoder that had merged unrelated color-9 regions. These are real learning
behaviors, not merely fluent narration.

**Then the goal remained unresolved.** After four actions it had restored the
moving structure to its starting position. Calls 36–40 repeatedly considered the
bottom-right color-8 marker as a destination. It tried a three-DOWN batch toward
that marker. The first DOWN left the structure unchanged and changed only two
bottom-bar pixels; the remaining two actions were skipped. The model recognized
this failure and proposed a wall constraint.

**The last two minutes produced no new game observation.** Its sixth and last
action, UP, arrived at 495.66 seconds. Calls 44–51 consumed another **121.47
seconds of model-call time** with no further action: inspecting candidate goals,
reconsidering the static dot, calculating a reachable-position graph, and
returning to the color-8-marker theory. The final helper could earn a matched
prediction, so lack of batch permission is not an adequate explanation for this
inaction.

The model said it would “brute-force explore” in call 47, but ran local movement
checks instead. Its BFS was conditional on its hypothesized collision model; it
was not evidence that all those positions were actually reachable. Likewise,
call 50 says “Context was evicted,” but that self-report alone does not establish
an actual eviction or its effect. Neither claim should be promoted to fact.

**Interpretation:** it could revise a mistaken causal model when the test was
clear. It was much less effective at selecting an action when several objective
theories remained plausible. More computation repeatedly revisited the choice
instead of reducing the uncertainty through contact with the game.

## 2. S5I5: a progress hypothesis became the objective

Evidence: [model calls][s-calls], [transitions][s-transitions],
[step log][s-notes], [prediction checks][s-checks].

**A reasonable opening.** Only CLICK and RESET were available. It identified
several objects and proposed testing whether clicking them changed them. The
first action arrived after 12 calls and 101.10 seconds. It checked the clicked
region and correctly observed that it had not changed; only the bottom row had.

**The unsupported promotion.** It considered different explanations for the
bar's one/two-pixel increments. In call 25 it nevertheless wrote:

> Confirmed: row 63 is the level progress bar

It had established that the bar changes with these actions, not that filling it
advances a level. Later notes call the goal provisional again, but behavior
continues to optimize bar filling. This is not consistently calibrated belief:
the label softens while the policy remains committed.

**The policy then narrows.** It built an encoder containing only the bottom row
and its count of color-4 cells, and a predictor adding one cell per click.
It retained one checked helper identity despite recurring two-cell increments.
After mismatches it submitted a single click to re-earn batching permission,
then another small batch. Calls 40 and 44–63 explicitly describe that strategy.
There were 31 matched, 12 mismatched and four unresolved prospective step checks.
Those counts describe predictor agreement—not success at the game.

**The transition census is decisive about what changed.** Of 47 executed
actions, **45 were CLICK 24 41**. Across **all 47 transitions there were zero
changed pixels outside row 63**. The bar ended with 60 color-4 pixels out of 64;
levels_completed remained zero. The final click was executed after the model
admission cutoff, within the reserved completion window.

Important limit: this run did **not** fill the bar completely. It therefore
does not falsify “a full bar causes a level transition,” nor establish that the
bar is a timer, life meter, or progress indicator. Calling it a death bar would
repeat the model's error with a different story. What is established is that
the policy committed almost all its executed actions to one unverified objective
proxy without distinguishing it from an action counter/resource interpretation.

**Interpretation:** the increased action rate is genuine, but it is not evidence
of increased useful work. The model optimized a measurable process while leaving
the reason that process should win untested. Its recognition that non-bar pixels
were unchanged (call 44) did not cause it to reopen the objective decision.

## 3. CD82: an unchanged predictor reinforced unsupported movement claims

Evidence: [model calls][c-calls], [transitions][c-transitions],
[host states][c-states], [prediction checks][c-checks], [generated helper][c-helper].

**Initial theory and revisions.** The model treated a color-15 component as a
player and a color-2 border as room walls. RIGHT and DOWN produced substantial
changes. The model noticed that the border moved/changed too, considered camera
scrolling, then an attached UI frame, then a static enclosing ring. These are
competing explanations, not established game mechanics. It did make exploratory
actions, including SPACEBAR, which changed pixels outside the bottom row.

**Four specific claims fail against the recorded pixels.**

| Observation followed by claim | Model's account | Actual complete pixel change |
| --- | --- | --- |
| Step 4, call 26 | LEFT moved the diamond by about one cell | (row 63, col 61), 4 -> 5 |
| Step 9, call 34 | RIGHT moved the diamond by one cell | (63,58), 4 -> 5 |
| Step 11, call 36 | Something happened on the clicked wall | (63,57), 4 -> 5 |
| Step 12, call 37 | UP worked; diamond moved up | (63,56), 4 -> 5 |

These are not merely uncertain interpretations: the claimed object's pixels did
not change. The host reports label=playfield and outcome=move for all four. In
[host_contract.py][host], any non-identical frame is classified this way; “move”
does not mean verified displacement of the model's nominated player. The model
read a broad change label as a specific causal result.

**A generated-code bug makes the agreement signal especially misleading.**
The final encoder emits player_rows, player_cols, player_center and player_count,
but does not emit a `player` key. Its predictor starts with:

```python
z2 = dict(z)
if z.get("player") is None:
    return z2
```

Consequently its directional movement code is never reached on those encoded
states. This is a bug in the model-generated helper, not a discovered game rule.
I verified the helper hash against the live check records, tested all four
directions on all 15 saved pre-states (**60 unchanged predictions**), and
reproduced all eight recorded checks using that exact helper identity.

All four bottom-bar-only transitions above received matched checks: the encoder
ignores the bar, so its player/border summary really was unchanged. Thus a matched
check and outcome=move could coexist while the model incorrectly claimed player
movement. The checker was correctly reporting agreement under that encoder;
it was not certifying the narrative. Feedback explicitly warns of that distinction.

This does **not** prove the helper bug explains the whole run. Even a working
one-cell displacement predictor would still require testing the object's identity,
the large shape changes, and the goal. It does establish a specific mismatch
between the model's narrated prediction and its executed predictor.

## Cross-game judgment

| Boundary | Established evidence | What it does not establish |
| --- | --- | --- |
| Getting to first contact | 13 / 12 / 9 calls before first LS20 / S5I5 / CD82 action; about 91 / 101 / 79 seconds | That all preparation was unnecessary |
| Learning mechanics | LS20 changed avatar/translation theory after contradictions | A complete or correct simulator |
| Choosing an objective | S5I5 committed to bar filling; LS20 kept reconsidering candidates | The actual goals of these games |
| Using predictions | CD82's actual helper predicted no change while its prose predicted movement | That prediction checks themselves should be removed |
| Responding to budget | Repeated “prioritize executable progress” comments; S5I5's late policy narrows to clicks | That the budget prompt caused the failures; there is no counterfactual arm |

The most important next distinction is **not more thinking versus more acting**.
LS20 needed a better choice of test; S5I5 needed to question what it was optimizing;
CD82 needed its claimed measurement and executed prediction to agree. A universal
“act faster” instruction could make the second failure worse.

The smallest useful follow-up checks would be: detect the CD82 predictor's
unchanged-output mismatch with its declared motion; ensure pixel-change feedback
is not read as player displacement; and require evidence separating an objective
proxy from mere action consumption before repeatedly pursuing it. These are
investigation targets, not implemented changes or demonstrated fixes for winning.

[ls-calls]: ../raw/qualification/solver-workspaces/ls20-9607627b-a9521i88/ipc/model_calls.jsonl
[ls-transitions]: ../raw/qualification/solver-workspaces/ls20-9607627b-a9521i88/ipc/transitions.jsonl
[ls-notes]: ../raw/qualification/solver-workspaces/ls20-9607627b-a9521i88/team_nemo/shared/knowledge/step_log.md
[ls-checks]: ../raw/qualification/solver-workspaces/ls20-9607627b-a9521i88/ipc/step_checks.jsonl
[s-calls]: ../raw/qualification/solver-workspaces/s5i5-18d95033-h5esvs0u/ipc/model_calls.jsonl
[s-transitions]: ../raw/qualification/solver-workspaces/s5i5-18d95033-h5esvs0u/ipc/transitions.jsonl
[s-notes]: ../raw/qualification/solver-workspaces/s5i5-18d95033-h5esvs0u/team_nemo/shared/knowledge/step_log.md
[s-checks]: ../raw/qualification/solver-workspaces/s5i5-18d95033-h5esvs0u/ipc/step_checks.jsonl
[c-calls]: ../raw/qualification/solver-workspaces/cd82-fb555c5d-7i8b7k_x/ipc/model_calls.jsonl
[c-transitions]: ../raw/qualification/solver-workspaces/cd82-fb555c5d-7i8b7k_x/ipc/transitions.jsonl
[c-states]: ../raw/qualification/solver-workspaces/cd82-fb555c5d-7i8b7k_x/ipc/states.jsonl
[c-checks]: ../raw/qualification/solver-workspaces/cd82-fb555c5d-7i8b7k_x/ipc/step_checks.jsonl
[c-helper]: ../raw/qualification/solver-workspaces/cd82-fb555c5d-7i8b7k_x/team_nemo/shared/helpers/world_model.py
[host]: ../runtime-source/chassis/host_contract.py
