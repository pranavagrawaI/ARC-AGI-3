# NOOA–Gemma epistemic-context A/B/C probe

## Question

Can Gemma abandon a self-generated game theory when authoritative successor evidence contradicts it, or does accumulated self-narrative dominate its next action?

This is a bounded mechanism probe, not a full-game evaluation.

## Frozen conditions

- Game: `ls20`
- Environment start: fresh offline game, default seed/state
- Model: `google/gemma-4-31b-it`
- OpenRouter endpoint: `modelrun/fp4` only; fallbacks disabled
- Endpoint preflight: ModelRun endpoint advertises image, reasoning, structured-output, and tool support with a 262,144-token context
- NOOA revision: `bd51d16e2594f0bc237b14fb31e9a880ec385b39` plus the preserved compatibility patch and this probe patch
- Agent variant: memory
- Skill: `interactive-game-solver`
- Visual representation: additive PNG plus hex grid
- Per-turn action cap: 20, deliberately retained so overconfident open-loop planning remains observable
- **Total environment-action cap: 44 per arm, enforced by `--max-env-steps 44`**
- Game-over reset budget: 0
- Harness fallback window: 86,400 seconds; a provider/model stall is manually terminated and classified incomplete rather than allowed to inject an action

## Arms

### A — control

Unmodified NOOA model context: full prior event history and ordinary memory behavior.

### B — evidence only

- prior CodeAct calls, generated prose, and generated-code comments are excluded from the next call;
- cross-turn model-authored memory is disabled;
- the current CodeAct call remains intact so tool results can be used;
- a deterministic host-owned ledger restores every executed action, pixel-difference count, reward, level-completion flag, and terminal state;
- the latest full state and programmatic `trajectory()` remain available.

This tests whether accumulated self-narrative causes the echo chamber.

### C — contradiction first

The full control history and memory remain. A dynamic host-owned block appears before event history and reports:

- changed versus no-op actions in the previous batch;
- reward and level-completion totals;
- the trailing identical-action no-op streak;
- an explicit requirement to classify the prior hypothesis `KEEP`, `REVISE`, or `ABANDON` before acting.

This tests whether contradiction salience is sufficient without deleting narrative history.

## Behavioral measures

Raw traces are primary. For each arm inspect:

1. first unsupported long batch;
2. first prediction/no-op contradiction;
3. actions until the policy changes;
4. repeated identical no-op actions;
5. whether the model distinguishes a local mechanism from the game objective;
6. whether it names a competing explanation and discriminating probe;
7. whether verbal acknowledgement changes the action policy;
8. environment actions, model calls, tokens, latency, levels, and stopping reason;
9. any rejected, truncated, fallback, or provider-failed action path.

A treatment does not pass merely because Gemma uses uncertainty language. Contradiction must alter its theory or next action.

## Executed runs

The first attempted A run exposed a harness defect: `max_env_steps` was checked
only between batches, so a batch starting at step 38 reached step 58. That run is
preserved but excluded with `INVALID_OVERSHOT_CAP.json`. The harness was repaired
to truncate each admitted batch to `max_env_steps - env_steps`, covered by focused
tests, and all three arms were rerun.

| Arm | Actions | Stop | Levels | Batches | Max batch | Calls | Input tokens | Pixel no-ops | Fallbacks |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A | 44 | action cap | 0 | 16 | 5 | 21 | 1,049,310 | 9 | 0 |
| B | 44 | action cap | 0 | 6 | 20 | 34 | 718,956 | 0 | 0 |
| C | 44 | action cap | 0 | 16 | 6 | 50 | 3,069,467 | 0 | 0 |

All three completed with `termination_reason=max_env_steps`, exactly 44 host
environment mutations, no harness fallback, and no REPL error. They are bounded
single-seed mechanism probes, not full-game evaluations. B had one zero-token
model call and recovered without fallback or environment mutation.

## What happened

### A — full-history control

A did not reproduce the 292-step runaway. It nevertheless repeated `UP` after a
falsifier: steps 24–29 were six consecutive exact pixel no-ops. It then changed
to `DOWN`. Thus this sample shows bounded contradiction latency, not sound
one-action falsification discipline.

### B — evidence-only history

Removing prior model narrative did not make the policy conservative. B submitted
only six batches, including a 30-action path that the existing per-turn guard cut
to 20, and later a 20-action batch. It repeatedly rebuilt a semantic player/goal
story from the current state and immediately promoted it into an open-loop plan.
Narrative deletion reduced accumulated input tokens, but did not solve unearned
action horizon.

### C — contradiction-first block

C never emitted any required `KEEP`, `REVISE`, or `ABANDON` status. More
importantly, it exposed a defect in the treatment's comparator:

1. At step 21, one `UP` moved Gemma's tracked player and color-12 wall by five rows.
2. The next 19 consecutive `UP` actions, through step 40, changed two pixels each
   but left both tracked component bounding boxes unchanged.
3. The host block therefore reported every action as “changed” rather than a
   no-op, because it used `diff_pixels > 0` instead of Gemma's committed predicted
   feature.
4. At step 26 Gemma saw five two-pixel outcomes and unchanged tracked entities,
   but attributed the movement from the preceding successful action to those five
   later actions. It then submitted three more `UP` macros.
5. At step 40 it finally acknowledged that the player and wall had stayed fixed
   and switched to `DOWN`—after 19 semantically ineffective repetitions.

C therefore did not “open its eyes.” It made the failure easier to audit, but the
advisory block neither owned the relevant prediction nor gated execution. It also
cost 50 calls and 3.07 million input tokens, substantially more than A or B.

## Evidence-licensed conclusion

No arm is a demonstrated cure. One sample per arm cannot separate treatment from
sampling variance, and the trajectories did not encounter identical
contradictions. The runs do establish three mechanism failures:

1. **History deletion is insufficient.** Gemma can regenerate an overconfident
   story from a single state and take a 20-action horizon.
2. **Any-pixel-change is not prediction success.** A two-pixel UI or resource
   change can conceal complete failure of the predicted entity transition.
3. **Advisory reconciliation is ignored.** A prompt requesting a status did not
   produce one, yet action authority remained available.
4. **Inspect-and-submit in one Python cell is not reconciliation.** Printed values
   are not seen by Gemma before later lines in that same cell submit actions;
   comments can therefore encode anticipated rather than observed output.

The next discriminating intervention should be a host-enforced two-phase gate:

```text
model commits prediction projection + expected value + falsifier
→ host admits one action
→ host computes authoritative successor and projection-specific verdict
→ model sees that verdict in a new call
→ host admits no next action until structured ABANDON/NARROW/RETAIN is valid
```

Macros should be unlocked only by repeated correct projection-specific forecasts,
and interrupted on the first mismatch. This is stricter than adding another
warning while remaining neutral about the game's theory.

`analyze_runs.py` reproduces the counts and writes `run-summary.json`. Its
model-tracked projection is explicitly scoped to this analysis: the largest
color-12 component and the 15-cell color-9 board component that Gemma itself used
in C. It is evidence for diagnosing this trace, not reusable solver theory.
