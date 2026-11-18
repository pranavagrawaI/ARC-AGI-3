# Discrimination replay — would the trigger have opened?

Offline replay of a recorded NOOA run. At every decision point it asks whether a
candidate discrimination trigger would have fired, using only artifacts already on
disk. Zero model spend.

Started 2026-09-05, after q4 spent 197 actions on ls20 level 0 while a hypothesis
marked `CONFIRMED (turn 1)` sat in its context refuting the hypothesis its actions
were organised around.

## What this establishes, and what it cannot

It establishes that a trigger fires **where** it should and **not everywhere**.

It cannot establish that firing helps. Term C of the trigger contract (an affordable
action exists whose outcomes separate the hypotheses) is only hinted at here, and the
whole downstream — probe chosen, belief revised, action policy changed — is invisible
to a replay of a run in which none of that happened.

> A positive replay is permission to run the live card. It is not a result.

## Why the substrate is `messages/`

`agent_logs/*/*/events.jsonl` records `repl_execute` rows carrying `code_length` and
`functions_used` — **not the cell source**. A replay built on the event log cannot see
a single thing the model wrote.

The recoverable record is `agent_logs/*/*/messages/step_NNN_round_RR_{user,assistant}.md`:
the `user` files are the exact rendered context at each decision point, the `assistant`
files the exact cells. Hypothesis state is parsed from the `stdout='''# Hypotheses …'''`
blocks in the `user` files, keyed to the step in the filename and forward-filled.

Three sources are joined:

| what | where |
|---|---|
| belief state as the model saw it | `agent_logs/*/*/messages/*_user.md` |
| action + rationale per turn | `agent_logs/*/*/events.jsonl` (`agent_turn`) |
| progress, level, availability | `ipc/states.jsonl`, `ipc/rejections.jsonl` |

## The two-sided contract

A trigger tuned until it fires on one failure trace is a detector for that trace. So
every evaluation runs both sides:

```bash
python3 replay.py --two-sided --trigger stall_repetition \
    --must-fire      <failing-run-dir> \
    --must-not-flood <succeeding-run-dir> \
    --before-step 188 --max-per-100 12
```

- **must-fire** — at least one opening before the deadline on the run that failed.
- **must-not-flood** — openings per 100 steps within budget on the run that succeeded.

Exit status is 0 only on `PASS`, so a launch script can gate on it the way cards gate
on `chassis-capability-audit/reachability.py`.

### VACUOUS is not PASS

If the control run cannot exercise the trigger — a belief-reading trigger against a run
with no belief stream — the verdict is `VACUOUS`, never `PASS`. A trigger that fails to
fire for want of input is indistinguishable from a specific one, and reporting that as a
pass is the same defect as a scorecard that reports three passes on a run that
accomplished nothing.

## Triggers

| name | reads beliefs | fires when |
|---|---|---|
| `stall_repetition` | no | no level in the window, actions spent past a floor, rationale repeating a recent turn above a similarity threshold, and at least one available action untried in the window |
| `hypothesis_conflict` | yes | a CONFIRMED hypothesis negates a term a live hypothesis still asserts, and the recent rationale stream still cites that term |
| `persistent_conflict` | yes | the same conflicting pair is still live, still contradictory and still cited N turns later |

Every opening prints both hypotheses and the overlapping token, so a reviewer can reject
a firing by hand. That auditability is the point: `hypothesis_conflict` is a heuristic
tuned on one trace and must be read as one. The first run of it produced a spurious
opening against `H13 (CONTRADICTED)` because the status lexicon did not treat
`CONTRADICTED` as retired — visible immediately in the output, fixed in the lexicon.

## Result: the conflict detector is falsified; the stall detector survives

Two failure traces (0 levels) against two success traces (1 level), separation measured
as openings per 100 steps on the failure divided by the same on the success.

| trigger | q4 fail | q2d success | ratio | verdict |
|---|---:|---:|---:|---|
| `hypothesis_conflict` | 2.5 | 11.4 | **0.22x** | FAIL |
| `persistent_conflict` | 2.0 | 7.3 | **0.28x** | FAIL |
| `stall_repetition` | 7.1 | 1.6 | 4.37x | PASS |

`stall_repetition` across all four pairings: q4/q2d 4.37x, q4/q3c 7.2x, q3b/q2d 3.49x,
q3b/q3c 5.74x.

**A confirmed hypothesis contradicting a live one is a signature of an active
investigator, not a stuck one.** q2d carried contradictions at four to five times q4's
rate *while clearing a level*. Adding a persistence term (the same pair still live,
still contradictory, still cited six turns later) does not rescue it: 0.28x.

This reverses the design that motivated the tool. The contradiction clock does not fire
selectively; the stall clock does. So the live discriminator should be **clocked by
stall**, with the current contradictions surfaced as material for choosing the probe
rather than as the trigger.

Claim ceiling: two failure traces and two success traces, all on ls20, n=1 per cell. A
0.22x ratio is strong enough to decline to freeze the conflict treatment. It is not
strong enough to call stall-clocking established.

## Pre-freeze evidence (`sweep.py`)

`stall_repetition`'s window, action floor and similarity cutoff were hand-chosen. Sweeping
3x3x3 settings across both failure and both success traces:

**27 of 27 settings separate in the correct direction**, ratio 2.37x to 17.38x. The
separation is a property of the signal, not of the thresholds. Choose the operating point
for review cost instead: `window=8, floor=25, sim=0.65` gives 17.4x at 4.3 openings per 100
steps; `window=4, floor=25, sim=0.55` gives 8.0x at 7.2 per 100. Fewer reviews cost less
and catch less. Precommit the choice rather than inheriting a default.

### Endpoint sizing — completion cannot be the gate at n=3

| trace | levels | stall events | recurrence within 20 steps |
|---|---:|---:|---:|
| q4 | 0 | 14 | 11/14 |
| q3b | 0 | 5 | 4/5 |
| q2d | 1 | 2 | 1/2 |
| q3c | 1 | 2 | 1/2 |

Two consequences for the live card.

**Completion is a coin flip on this chassis family.** 2 of 4 runs cleared a level, p≈0.5.
At n=3 per arm, even a treatment lifting completion to 0.9 gives a 0.09 chance of an
unambiguous 0/3-vs-3/3 split, and `P(treatment > control)` moves only from 0.34 under the
null to 0.76. Completion is a reportable secondary, not the gate.

**Stall events are the well-powered unit.** ~10 per failing run, ~2 per succeeding run, so
a 3-run arm yields roughly 6–28 reviewable events — enough to estimate an escape or
recurrence rate with real precision.

**Recommitment failure already has a control baseline.** With no intervention at all, a
stall recurs within 20 steps in 11/14 and 4/5 of the failure traces' events (~79%), against
1/2 and 1/2 in the success traces. The treatment has a number to beat before anything is
run.

## Rediscovery baseline (`rediscovery.py`) — the endpoint fails too

Written to size a game-local-skill card, since level-1 completion cannot gate one:
level 1 has been entered 3 times and completed 0 times. Three channels, over the
303 level-1 actions in q2d, q3a and q3c.

| channel | what it counts | pooled |
|---|---|---:|
| R3 | length-1 probes in L1 re-confirming a settled L0 action law | **0.030** |
| R3s | re-testing an action base that was inert throughout L0 | **0.000** |
| R2 | helper functions defined in L0 and redefined in L1 | **0.000** |
| R1 | L1 authored claims restating an L0 claim (token overlap ≥0.6 / 0.75 / 0.9) | 0.183 / 0.049 / 0.014 |

R3 has a floor effect: only 17 of 303 level-1 actions are length-1 probes at all.
The actors do not explore level 1, they execute routes. R2 is a true zero — q2d
defined `encode`/`predict`/`retrodict` and q3a six functions in level 0, and
neither redefined any of them in level 1. R1's loose threshold is over-counting;
inspection shows the hits are level-1-specific facts sharing vocabulary
("Tile at (15,34). RIGHT blocked by walls…"), and at 0.9 it is 0.014.

**Conclusion: rediscovery is not the failure mode, so it cannot be the endpoint.**
Nothing is being re-derived. If anything the corpus shows the opposite — q3c
carried its level-0 predicate into level 1 *too* strongly and defended it behind
19/19 retrodiction. The gap is not transfer, which is already near-total. It is
revocation. That is q6's P12, and a game-local-skill card should be measured on
whether a seeded skill's defeater fires, not on whether it prevents repetition
that does not occur.

Method note: helper source is passed as a variable (`src = '''…'''` then
`write_helper("game.py", src)`), so an AST walk that only accepts literal
arguments reports zero helper writes. The first version of this tool did exactly
that. Unresolved writes are now counted and printed rather than silently dropped.

## Integrity: the corruption is in the render path

`--integrity` diffs AUTHORED (recovered from cell source) against STORED (`knowledge/`)
against RENDERED (last readback). On q4:

```
AUTHORED 34 hypotheses from 15 note writes
STORED   34 in knowledge/hypotheses.md
RENDERED 15 in the last readback
6 semantic mutation(s) of actor-authored belief  -> INTEGRITY FAIL
```

Every mutation is `/X` -> `/workspaceX`, and **AUTHORED -> STORED is clean**. So
`append_notes` writes faithfully and the corruption enters between reading the file and
placing it in the prompt. H7 reaches the model as
`decorations/workspacegoals`; the agent wrote `decorations/goals`.

Mechanism still unidentified. `_build_redactions`' parent walk does emit a bare-slash
replacement, but only for run dirs three levels deep or less, and this one is deeper —
tested and refuted.

## The traces

All ls20. `levels` is the scoring outcome; `authored` counts hypotheses recovered from
cell source; `readbacks` counts snapshots the agent actually re-read into context.

| trace | levels | steps | authored | readbacks | role |
|---|---:|---:|---:|---:|---|
| `20260904_165006_…_q4` | 0 | 197 | 34 | 36 | must-fire |
| `20260904_035318_…_q3b` | 0 | 89 | 0 (unparsed format) | 0 | must-fire, stall only |
| `20260903_160733_…_q2d` | 1 | 123 | 30 | 16 | must-not-flood, non-vacuous |
| `20260904_140403_…_q3c` | 1 | 202 | 2 | 0 | must-not-flood, stall only |

`stall_repetition`'s two q3c firings land at steps 186 and 198, after level 0 was cleared,
while it was genuinely stalled on level 1 — arguably true positives on a second stall
rather than false alarms.

### Belief upkeep does not track success

q4 authored 34 hypotheses and re-read them 36 times, and completed nothing. q3c authored
2, left both UNTESTED, never re-read them, and cleared a level. q2d authored 30, cleared a
level. Whatever separates these runs, it is not the quantity or the upkeep of recorded
belief — which is a second reason, independent of the falsification above, to stop looking
for the remedy in memory machinery.

### The readback floor, now closed

Rendered snapshots begin at step 43 — the first time a q4 cell read the file back. The
authored clock recovers writes from cell source and starts at step 0, so the H7/H2
contradiction is now detected at **step 1, turn 1** rather than step 69. Readback lag was
43 steps, a fifth of the run's budget.

Availability was therefore never the bottleneck, which retires the wording we had been
using. The defensible claim is not that the read path is absent but that **a confirmed
contradiction had no governing authority over action selection** — and the falsification
above says arbitration keyed on the contradiction itself will not find the moment.

### Note format is model-chosen, not schema-enforced

q4 and q3c wrote `## H7: claim` headings with `- Status:` lines. q2d wrote a markdown
table. q3b wrote a third shape this parser still does not read (10 note writes, 0
hypotheses recovered). Anything that must read these notes has to tolerate all of them.

That is a direct argument for giving the live discriminator a **typed object** rather than
letting it parse free-form markdown: a trigger whose sensitivity depends on which format
the model happened to pick is not measuring what it claims to.

## Standing rule

Replay a candidate trigger two-sided before spending a live run on it. A trigger that
cannot be shown to fire on q4 and stay quiet on a success trace is not ready for a card,
and a `VACUOUS` control means the specificity question has not been asked yet.
