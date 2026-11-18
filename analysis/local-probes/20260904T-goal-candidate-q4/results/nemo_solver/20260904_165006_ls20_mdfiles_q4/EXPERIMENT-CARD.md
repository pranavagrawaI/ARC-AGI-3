# q4-goal-candidate-record

Written 2026-09-04 after q3c scored PASS, before any q4 implementation.
Frozen at launch. Predecessors: `../20260904T-host-control-loop-q3/`
`EXPERIMENT-CARD.md` (q3a), `-q3b.md`, `-q3c.md`, and q3c's `RESULT.md`.

Question: q3c's scribe produced a model-written goal predicate and the actor
*used* it. On the next level the predicate was wrong, and the actor defended it
for the rest of the run behind 19/19 dynamics. Does a **host-owned** candidate
record — predicted terminal signature, cheapest discriminator the host can
score, deadline in env actions — make that predicate get re-tested instead of
defended?

Posture: targeted repair of the failure q3c named. Not a new control-loop
bundle, not a dynamics card, not a thinking-on card.

Contrast: q3c, `../20260904T-host-control-loop-q3/results/nemo_solver/20260904_140403_ls20_mdfiles_q3c/`.
Held fixed: game `ls20`, variant `mdfiles`, model `qwen/qwen3.8-27b`, skill
`interactive-game-solver`, `--visual off`, `--effort-ladder ''`, thinking off,
sandbox `inproc --sandbox-cells --sandbox-require --sandbox-cpu-s 60`,
`--agent-turn-timeout 600`, `--max-turns 60`, `--max-env-steps 200`, the q3c
scribe, and every q3a/q3b/q3c gate except the one named below.

Changed, one thing: the goal-candidate record and the host scoring around it.

---

## Why this, and why not more machinery

q3c cleared level 0 in 32 actions, wrote `level_0_insights.md` itself (P2 =
1.000, zero scribe rejections), and transferred the predicate *"Items unlock
areas. Goal is in a restricted area requiring items."* Level-1 retrodiction was
19/19, 16/16, 19/19. It did not clear the level.

The L1 batches were searched routes under that predicate (t15 n=17 to items,
t17 n=19 to items, t18 n=20 to the goal after pickup, t22 n=19 pickup-as-test,
t33 n=18 back to items). First explicit doubt is t28. Then t33–t37 return to
the same theory. Every L1 `predicted_path` `expect` was `"move"`.

Two consequences, both already in the ledger:

1. Dynamics is not the constraint. Frontier 4 is. (Board 119; ledger causal
   map: *objective-discriminating experiment — recurrent weak link*; Q2.)
2. A discriminator that is only a dynamics probe will **confirm** the stale
   goal. q3c's 19/19 would have promoted it. The discriminator has to predict a
   *terminal-class* outcome (`level_up` or `terminal`) or it is not a test of
   the objective.

Ledger Q2 asks the question this card is for: would requiring a discriminator
improve action choice, or merely create another ritualized report? The
treatment is shaped to fail if the answer is ritual.

Tycho already fires a builder at `LEVEL_START` to check whether
`init_state`/`outcome` still hold (`dispatcher.py:level_start_reason`). That is
a world-model repair pass. It is **not** this card. Check-Tycho-first (q3c
decision 2) is satisfied by reading that function and declining to port it:
q3c's helper was already 19/19 on the new level. The missing obligation is
objective-side.

---

## Treatment (frozen)

Host-owned record at `ipc/goal_candidate.json`. Not a markdown file. The model
does not become right by writing a schema.

```text
hypothesis          model-written string; optional for scoring
predicted_terminal  one expect in {level_up, terminal}
discriminator       a predicted_path of length 1–5 containing at least one
                    expect in {level_up, terminal}. Longer is a plan, not bound.
deadline_actions    host-set; see below
status              host-set: proposed | tested | contradicted | expired | promoted
score               host-set; the discriminator's scored predicted_path
```

Activation: on `levels_completed` increase, after the scribe evidence is
written, the host opens a candidate with `status=proposed`,
`discriminator=null`, and

```text
deadline_actions = max(16, actions_spent_on_the_level_just_finished)
```

On q3c that is `max(16, 32) = 32` env actions into level 1. If the transferred
theory were true, a route of similar length should hit a terminal. The floor
stops a lucky 5-action finish from expiring the next level immediately. The
number is observational (a count the host already stores in
`level_*_terminal.json`), not a named object role.

Assist, not refusal — q3a vs q3c is settled. Missing discriminator does not
reject `submit_actions`. Teeth are the deadline and the status line, the same
shape as `SCRIBE PENDING`:

```text
CANDIDATE UNTESTED: level L hypothesis is proposed;
  N actions remain to submit a discriminator whose expect includes
  level_up or terminal. self.goal_candidate() is the record.
CANDIDATE CONTRADICTED: discriminator scored, terminal expect missed.
  Do not use as governing objective.
CANDIDATE EXPIRED: deadline reached with no scored discriminator.
```

`self.goal_candidate()` returns the live record. The model fills `hypothesis`,
`predicted_terminal`, and `discriminator` by calling it with those fields, or
by submitting a `predicted_path` that contains a terminal-class expect — the
host then treats that batch as the discriminator if none is on file. A
discriminator whose expects are all `move`/`no_impact` is structurally
incomplete and is not accepted as the discriminator (play still proceeds).

Host scoring reuses `score_predicted_path`. After the discriminator batch
executes:

- terminal-class expect matched → `status=promoted` (and if the level
  actually completed, that is also a P8 observation)
- terminal-class expect missed → `status=contradicted`
- deadline reached with `discriminator` still null or never executed →
  `status=expired`

Host fallback at deadline, analogous to `_write_consolidation_fallback`: write
the expired record with reason `no discriminator submitted`. That is a P6
failure, not a save. Marker: `"Deterministic fallback candidate:"`.

Status line also shows the only new gauge: `actions_this_level=A/deadline=D`.
No other constraint gauges (wall, context %, max_turns remaining) — those are
a different card.

Contestant prior holds: the host names outcome classes it already computes
(`level_up`, `terminal`, `move`, `no_impact`), a count, and a deadline. It
does not name objects, coordinates, keys, boxes, or win conditions.

### Deliberately not in this change set

- Tycho's `LEVEL_START` trigger-builder. Dynamics already transfer.
- A `goal_candidate.md` / insights schema. That is the ritual Q2 named.
- Submit-rejection of play for an untested or expired candidate. Assist
  settled that.
- Helper rewrite (P5). Still recorded, not gated.
- Host check for helper-constant vs confirmed-note contradiction (q3b's
  inverted wall map). Own card.
- Context-budget repair as treatment. Pre-launch check only; q3c already ran
  at ~56k mean prompt tokens and every tempo claim in that card is void.
- Thinking-on. Own card, own hatch.

---

## Prediction

| # | metric | q3c | q4 predicted |
|---|---|---:|---|
| P1 | batches >1 whose `predicted_path` scored ≥0.6 | 0.944 | **≥ 0.50** (hold) |
| P2 | completed levels with a **model-written** consolidation | 1.000 | **= 1.00** (hold) |
| P3 | re-entry rate over attempted actions | 0.020 | **≤ 0.05** (hold) |
| P4 | waste share of executed env actions | 0.064 | **≤ 0.15** (hold) |
| P6 | of level-transfers reached, fraction whose discriminator (terminal-class expect, executed, host-scored) landed before `deadline_actions` | 0.000 | **= 1.00** |
| P7 | after first `contradicted` or `expired` on a level: a *different* discriminator action-tuple is submitted within one more deadline, **or** every subsequent batch until then has length 1 | n/a (no discriminator existed) | **= 1.00** |

P1–P4 are hold-the-line. The failure worth catching is a frontier-4 ritual
that displaces the control loop q3c just earned.

P6 is the severity clause. A written-but-unexecuted discriminator, a
dynamics-only discriminator (all `move`), a host fallback, or a long plan
that happens to end in `level_up`, is a miss. q3c's incumbent is 0: its L1
terminal-class batches were length 11/20/17.

P7 is the anti-defense clause. It only runs if a discriminator was scored
and missed, or expired. q3c after an analogue of t22 (pickup-as-test, no
level-up) kept submitting n=19/14/18 routes of the same theory — that is
the behavior P7 has to fail. Length-1 probes and RESET count as revision
attempts; a second discriminator must differ in its action-tuple.

P5 (helper rewrite) and P8 (predicted terminal signature scored against an
actual later completion) are **recorded, not gated**. P8 is unmeasurable
unless a second level completes; do not fail the run for that.

RHAE is not the gate. 200 env steps still caps the reachable score near 0.10.

---

## Measure

`score_q4.py`, written and frozen before launch, in this directory.
`score_q3b.py` is left exactly as it scored q3c and is not reused.

The scorer reads only run artifacts and recomputes labels from frames, as
`score_q3b.py` does. It must not trust `ipc/goal_candidate.json`'s `status`
field for a pass:

- P1, P3, P4: same as `score_q3b.py`
- P2: same; `Deterministic fallback note:` is a P2 miss
- P6: for each completed-level transfer, a batch submitted before
  `deadline_actions` env steps on the new level has a `predicted_path`
  containing `level_up` or `terminal`, that batch executed, and the scorer's
  own `score_predicted_path` against frames is defined. Host fallback marker
  `Deterministic fallback candidate:` is a P6 miss
- P7: action-tuples from `ipc/actions.jsonl`; length from the same; T* is the
  env step of the first missed terminal-class discriminator or of deadline
  expiry, recomputed from frames and the deadline formula, not from host status

Also reported, not gates: P5, P8, force-advances, input tokens / env action,
`<truncated>` marker count, candidate status transitions, wall, spend.

---

## Cap

60 agent turns, 200 env steps, `--agent-turn-timeout 600`, 4 h wall, **$6**
OpenRouter (q3c: $3.68 for 38 turns / 202 env actions). Pre-launch: confirm
remaining credit ≥ $6. Do not log the key.

---

## Stop

- Score after ≥15 agent turns and at least one level-transfer, or after
  cap if no transfer occurred.
- Abort on spend ≥ $6, wall ≥ 4 h, `--max-env-steps` reached.
- Refusal deadlock: ≥8 consecutive rejected submits (`ipc/ABORT`).
- Abort if `reasoning_tokens > 0`.

---

## Invalid if

- `reasoning_tokens` return; PDEATHSIG kills the orchestrator; fewer than 15
  agent turns; game id in an emitted cell; a gate throws and is silently
  bypassed.
- The discriminator is accepted from markdown, or `ipc/goal_candidate.json`
  is absent after a level-transfer — the port did not fire and P6 measures
  nothing.
- A dynamics-only path (no terminal-class expect) is counted as a
  discriminator by the scorer. That would grade q3c's 19/19 as a pass.
- **No level completed** — P6/P7 INCONCLUSIVE. See result map: that outcome
  ends this line, it does not license a q4b that only raises the budget.

The known-broken context budget is **not** an invalidity. q3c already ran
under it. Do not draw tempo claims. See pre-launch.

---

## Result map

| result | next action |
|---|---|
| **P6 = 1.00, P7 = 1.00, P1–P4 hold** | The record is not a ritual: a terminal-class probe ran, and a miss was not defended as an assertion. Retain. Next is the thinking-on card **on frontier 4** (q3c decision 4), or a depth run scored on RHAE, each with its own card. Not more dynamics machinery. |
| **P6 = 1.00, P7 = 0** | They ran the test and kept the stale objective anyway. The record reports; it does not govern. Do not add fields. Next is host teeth on *governing* use of a contradicted candidate (still assist-shaped: status line + refuse *long batches that resubmit the missed discriminator action-tuple*), own card. |
| **P6 = 0 via fallback / never filed** | The model would not produce a scorable terminal-class probe. Keep the fallback. This is the strongest remaining case for the thinking-on card, which then becomes next. |
| **P6 = 0 because a markdown schema was filled and never executed, or because the accepted discriminator was all `move`** | We built the ritual Q2 named. **Delete the schema.** Keep only: a `predicted_path` with a terminal-class expect is the discriminator, deadline expiry marks expired. Re-run that deletion as q4-del, new card. |
| **P1 or P2 collapse, P6 met** | Frontier 4 displaced the control loop or the scribe. Restore the q3c byte snapshot from `patched-source/after/` and stop adding. |
| **P6/P7 INCONCLUSIVE (no level)** | **Stop running this experiment.** q3a reached a boundary, q3c reached a boundary, a third miss means the question is again "why can this actor not find a goal", which q3b already answered and q3c showed is not a dynamics problem. Different card, not a larger budget. |
| **Any Invalid-if** | Do not interpret. New id, new card. Preserve the run. |

---

## Outputs

`analysis/local-probes/20260904T-goal-candidate-q4/results/nemo_solver/<timestamp>_ls20_mdfiles_q4/`
— as q3c, plus `ipc/goal_candidate.json` (append-only copies per status
transition permitted; the live file is the current record). Scored by
`score_q4.py` in this directory.

---

## Pre-launch, required, not part of the treatment

1. **Snapshot.** q3c's live sources are uncommitted in
   `~/projects/arc-agi-3-reference/nooa/examples/arc_agi_3/` and were not in
   `experiments/agents/0158-.../source-snapshots/nooa/` (that tree has no
   `harness.py`). Copies now sit in
   `../20260904T-host-control-loop-q3/patched-source/after/`. Implement q4
   against those bytes. Do not overwrite the 0158 snapshot as a side effect
   of this run; promotion is a separate decision.
2. **Context budget check, before any paid call.** q3c set
   `max_context_tokens=32_000` and left `max_event_tokens=None`; mean prompt
   tokens 55,883 and `<truncated>` in zero cells. Set `max_event_tokens`,
   run a scripted/offline agent far enough to render history, and confirm
   the marker appears. If it still does not, ship q4 anyway at the q3c
   prompt size and write that in the RESULT; do not add a third truncation
   knob as treatment.
3. Launch pattern unchanged from q3c: `daemonize_*.py` double-fork
   (PDEATHSIG), bash `set -a` on `~/projects/arc-agi-3/.env`,
   `--effort-ladder ''`, model id `qwen/qwen3.8-27b`, `ARC_DATA_DIR` =
   worktree root. Watch `ipc/states.jsonl` notes, not `q4*.out`. File checks
   in Python, not shell globs.

---

## Pre-launch amendment (2026-09-04, before any execution)

Recorded because scoring the q3c incumbent with the card-as-written made P6
pass. q3c's L1 batches at t16/t18/t19 carried a `level_up` expect on the
*last* of 11/20/17 actions — searched routes under the transferred goal, not
probes. Without a length bound those are indistinguishable from a
discriminator, and the run cannot fail.

**A discriminator batch has length 1–5** and contains a terminal-class expect.
Longer terminal-class paths are plans; they play, they are not bound, they do
not score P6. Host, agent, and scorer share `DISCRIMINATOR_MAX_ACTIONS = 5`.
q3c under this rule: P6 = 0, P7 = 0.

This does not change the question, contrast, or stop conditions other than
the spend cap below.

**Spend cap $8 → $6**, authorized 2026-09-04 before launch. Remaining credit
was $6.32. q3c cost $3.68 at the same turn/step budget. Watcher aborts at $6.

---

## Standing decisions this card does not reopen

1. Assist beats refusal at a lifecycle boundary (q3a vs q3c).
2. Check Tycho before designing NOOA lifecycle machinery. `LEVEL_START`
   trigger-builder was checked and is not this mechanism.
3. RHAE is not a gate at these budgets.
4. Thinking-on remains its own card and its own hatch.
5. `--agent-turn-timeout 600` retained.
6. WikiSkill, latent bridge, V16 re-push: still not next.
7. `score_q2.py`, `score_q3.py`, `score_q3b.py` left as they ran.
