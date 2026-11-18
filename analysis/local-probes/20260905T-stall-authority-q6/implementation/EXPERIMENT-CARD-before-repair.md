# q6-stall-authority-review

Written 2026-09-05, before any q6 implementation. Frozen at launch.
Predecessors: `../20260904T-goal-candidate-q4/EXPERIMENT-CARD.md` and its
`RESULT.md`; `../20260904T-host-control-loop-q3/EXPERIMENT-CARD-q3c.md`.
Offline evidence this card rests on: `../../discrimination-replay/README.md`,
`replay.py`, `sweep.py`.

Question: q4 spent 197 env actions on ls20 level 0 while re-issuing variations of
one policy. Does a **stall-clocked authority review** — suspend the current
policy, force the Actor to expose and either recommit to or replace the belief
that is governing, act once, and record whether authority changed — make stalls
end, instead of recur?

Posture: targeted repair of the failure q4 exhibited. Not a search card, not a
conflict-arbitration card, not a memory card.

Contrast: one chassis, one flag. `ARC_STALL_REVIEW=0` (control) and
`ARC_STALL_REVIEW=1` (treatment). 3 runs per arm, launch order interleaved
`C T C T C T`. Held fixed across all six: game `ls20`, variant `mdfiles`, model
`qwen/qwen3.8-27b`, skill `interactive-game-solver`, `--visual off`,
`--effort-ladder ''`, thinking off, sandbox
`inproc --sandbox-cells --sandbox-require --sandbox-cpu-s 60`,
`--agent-turn-timeout 600`, `--max-turns 60`, `--max-env-steps 200`, the q3c
scribe, and every q3a/q3b/q3c/q4 gate.

Changed, one thing: whether a detected stall suspends play for one authority
review.

---

## Why this, and why not the mechanism we were going to build

Two exchanges ago this card was going to be a **conflict** discriminator. The
offline replay falsified that before a dollar was spent.

| trigger | q4 (0 levels) | q2d (1 level) | ratio |
|---|---:|---:|---:|
| `hypothesis_conflict` | 2.5 /100 | 11.4 /100 | **0.22x** |
| `persistent_conflict` | 2.0 /100 | 7.3 /100 | **0.28x** |
| `stall_repetition` | 7.1 /100 | 1.6 /100 | **4.37x** |

The **successful** run carries contradictions four to five times more densely
than the failing one. A confirmed hypothesis contradicting a live one is the
signature of an active investigator, not a stuck one. Adding a persistence term
does not rescue it.

What separates the arms is stall: no progress, rationale repeating, options
untried. That separation holds across all four pairings of two failure traces
against two success traces (4.37x, 7.2x, 3.49x, 5.74x) and at **27 of 27**
threshold settings in `sweep.py` (2.37x–17.38x), so it is a property of the
signal rather than of a tuned cutoff.

Therefore: contradictions are **material supplied to the review**, never the
interrupt. The interrupt is stall. The thing under test is whether an agent can
be made to consciously recommit to, or abandon, a policy that has stopped
paying — not whether it can be made to notice a contradiction, which it already
does constantly and harmlessly.

### Deliberately not in this change set

- Bounded search (`plan_bfs`/`plan_astar`). Untouched until this reports.
- q5 symmetry seeding. Launched 2026-09-04 22:30, torn down by SIGTERM, no
  RESULT. It is paused, not superseded, and is not this card.
- Conflict or persistence triggers. Falsified above; they stay in `replay.py`
  as recorded negatives.
- Any parsing of hypothesis IDs out of markdown. q2d proved the note format is
  model-chosen, not a contract (headings, tables, and a third shape `replay.py`
  still cannot read). The review authors its basis fields at intervention time.
- Refusal of `submit_actions`. Assist beats refusal is settled (q3a vs q3c).
- Root-causing the render-path mutation. Contained by assertion, not fixed.

---

## Treatment (frozen)

**Trigger**, host-owned, no model call. Operating point precommitted as
`window=6, min_actions=25, similarity=0.55` — the middle of the swept range
(4.89x separation) and the setting every baseline number below was computed at.
A stall is detected at an agent turn when all four hold:

```text
levels_completed unchanged over the last 6 agent turns
env actions spent >= 25
this turn's rationale repeats a rationale from those 6 turns at
  token-set Jaccard >= 0.55
>= 1 available action untried in those 6 turns
```

**On detection the current policy is suspended.** The turn does not end with an
ordinary `submit_actions`. The Actor is handed the stall evidence — the repeated
rationales, the untried actions, the current hypotheses as it last authored them
— and must return exactly one:

```python
class StallResolution(BaseModel):
    disposition: Literal["continue", "probe", "switch"]
    previous_basis: list[str]     # authored now, not parsed from markdown
    active_basis: list[str]       # what governs the next action
    action: str                   # one legal action; batch length 1
    abandon_if: str               # required for `continue`
    predictions: dict[str, str] | None   # required for `probe`
    rationale: str
```

Host validation, structural only. The host never rules on whether a belief is
true:

- `continue` requires non-empty `abandon_if`. Without a stated falsifier,
  continuation is the ritual this card exists to catch.
- `probe` requires `predictions` with >= 2 distinct predicted outcomes.
- `switch` requires `active_basis != previous_basis`.
- Any disposition: `action` legal from the current state, batch length 1.
- A malformed resolution returns one `"REJECTED: ..."` string and may be
  repaired in the same turn, as `submit_actions` does today.

**Four-event trace**, `ipc/authority.jsonl`, append-only:

```text
STALL_DETECTED        step, turn, thresholds, repeated rationales, untried actions
AUTHORITY_REVIEWED    disposition, previous_basis, active_basis, abandon_if, predictions
CONSEQUENCE_OBSERVED  action, predicted, actual, abandon_if_satisfied (bool|null)
AUTHORITY_RESOLVED    retained|changed, active_basis_after, stall_escaped (bool, at K=20)
```

`abandon_if_satisfied` is the closing link. Without it, `continue` with a
beautiful falsifier is indistinguishable from `continue` with a decorative one.
It is evaluated by the host against the observation using only classes the host
already computes (`move`, `no_impact`, `level_up`, `terminal`, grid-hash change,
`levels_completed`); if the stated condition is not expressible in those terms
the host records `null` and that is a P12 miss, not a crash.

Contestant prior holds. The host names a stall (a count, a repetition measure, a
set difference over available actions), demands a typed answer, and records what
happened. It names no object, coordinate, goal, or win condition.

## Control (frozen)

**The detector is armed in both arms.** The control logs `STALL_DETECTED` and
plays on; no review, no suspension, no extra model call. This is what makes the
primary endpoint paired: both arms produce the same event stream under the same
rule, and only the response differs. A control with the detector off would leave
P9 with no denominator.

---

## Prediction

Baselines are the offline replay's control estimate over 23 stall events in the
four historical traces (`sweep.py`).

| # | metric | baseline | q6 predicted |
|---|---|---:|---|
| **P9** | **stall escape rate — fraction of `STALL_DETECTED` events with no further detection within K=20 env steps** | **0.26 pooled (0.21 on the two 0-level traces)** | **complete separation: every treatment run's rate exceeds every control run's** |
| P9b | stall incidence per 100 env steps | 6.4 | recorded; a treatment that *prevents* stalls is success and must not be invisible |
| P10 | recommitment failure — `continue` → same `active_basis` → stall recurs within K | n/a (no review exists) | **<= 0.50** of `continue` dispositions |
| P11 | review latency, actions from detection to `AUTHORITY_REVIEWED` | n/a | **= 0** by construction; any non-zero is an implementation fault |
| P12 | of `CONSEQUENCE_OBSERVED` rows where `abandon_if_satisfied=true`, fraction with `AUTHORITY_RESOLVED=changed` | n/a | **= 1.00** |
| P1 | batches >1 whose `predicted_path` scored >= 0.6 | 0.944 / 0.949 | **>= 0.50** (hold) |
| P2 | completed levels with a model-written consolidation | 1.000 | **= 1.00** (hold) |
| P3 | re-entry rate | 0.020 / 0.033 | **<= 0.05** (hold) |
| P4 | waste share | 0.064 / 0.036 | **<= 0.15** (hold) |

**P9 is the gate.** The test is distribution-free: with 3 runs per arm, complete
separation of the six per-run rates occurs by chance with probability
1/C(6,3) = **0.05**. That is the whole reason the arm size is 3 and the reason
the endpoint is per-event rather than per-run outcome.

A run contributes to the separation test only with **>= 3 stall events**. If
fewer than 3 runs per arm qualify, P9 is INCONCLUSIVE and the pooled event-level
rate is reported with that caveat — never as a pass. A run that completed levels
and barely stalled is good news and is recorded under P9b and completion, not
discarded.

**Completion and RHAE are reported, never gated.** Base rate on this chassis
family is 3 of 5 substantive runs clearing level 0 (q2d, q3a, q3c; not q3b, q4),
p~0.6 — corrected 2026-09-05 from the p~0.5 of the four-trace replay corpus,
which omitted q3a. The correction makes completion a slightly *worse* gate, not
a better one. **Level 1 has been entered 3 times and completed 0 times; no run
has ever reached level 2.** At n=3 per arm, a treatment
lifting completion to 0.9 still gives only a 0.09 chance of an unambiguous
0/3-vs-3/3 split, and `P(treatment > control)` moves from 0.34 under the null to
0.76. Completion cannot carry a gate at this arm size and this card does not
pretend otherwise.

P10 is the anti-ritual clause, and it is the branch to watch. If treated stalls
recur at the historical 0.79 while the review dutifully emits well-formed
objects, we will have built another P1-shaped ceremony and the mechanism dies
even if P9 passes.

---

## Measure

`score_q6.py`, written and frozen in this directory **before launch**. Prior
scorers are left exactly as they ran.

The scorer reads only run artifacts and recomputes labels from frames. It must
not trust `ipc/authority.jsonl` verdict fields for a pass:

- **P9 / P9b**: recompute `STALL_DETECTED` from `events.jsonl` rationales and
  `ipc/states.jsonl` progress using `replay.trigger_stall_repetition` at the
  frozen operating point. Escape is recomputed from that recomputed event list,
  not read from `AUTHORITY_RESOLVED.stall_escaped`.
- **P10**: `continue` rows joined to the next recomputed detection; `active_basis`
  compared verbatim as authored.
- **P11**: env-step delta between recomputed detection and the review row.
- **P12**: `abandon_if_satisfied` recomputed from frames using the host's own
  outcome classes; rows the host recorded `null` count as misses.
- **P1–P4**: as `score_q4.py`.

Also reported, not gated: disposition frequencies, authority-change latency
(actions from an `abandon_if_satisfied=true` to the first changed
`active_basis`), levels, RHAE, force-advances, `<truncated>` count, wall, spend.

Cross-arm integrity: `replay.py --integrity` must pass on every run.

---

## Cap

Six runs. Per run: 60 agent turns, 200 env steps, `--agent-turn-timeout 600`,
4 h wall. Treatment runs additionally cap **review calls at 20 per run**; on the
21st detection the host logs `STALL_DETECTED` and plays on, as the control does.

Spend: **$5 per run, $30 total.** q3c cost $3.68, q4 $3.56 at the same
turn/step budget; treatment adds ~10 review calls per run. Watcher aborts the
arm at $30 cumulative.

Wall: 6 x ~95 min sequential, ~10 h. Concurrency is not authorized — the single
local model server is the constraint that made every prior card sequential.

**Pre-launch credit gate: remaining >= $30.** Last recorded remaining was
**$0.16** (`../20260904T-symmetry-tool-use/results/20260904_233400/spend-start.json`,
2026-09-04). This card cannot launch without a top-up. Do not start a partial
arm.

---

## Stop

- Score the pair after all six runs complete, or after any arm-level abort.
- Abort the arm on cumulative spend >= $30 or total wall >= 14 h.
- Abort a run on `--max-env-steps`, spend >= $5, wall >= 4 h.
- Refusal deadlock: >= 8 consecutive rejected `StallResolution` submissions
  (`ipc/ABORT`).
- Abort any run on `reasoning_tokens > 0`.
- Abort the arm if `replay.py --integrity` fails on a completed run: the Actor
  did not receive the belief it authored, and no belief-governance claim is
  interpretable. Fix the render path, new run id, new card.

---

## Invalid if

- `reasoning_tokens` return; PDEATHSIG kills an orchestrator; a run has fewer
  than 15 agent turns; a game id appears in an emitted cell; a gate throws and
  is silently bypassed.
- The two arms differ in anything but `ARC_STALL_REVIEW`. Byte-compare the
  chassis before each launch.
- The control arm emits no `STALL_DETECTED` rows — the detector was not armed
  and P9 has no denominator.
- `previous_basis` / `active_basis` are populated by parsing `knowledge/*.md`
  rather than authored at intervention time. That reintroduces the format
  dependence q2d exposed and P12 would measure the parser, not the agent.
- The scorer reads `AUTHORITY_RESOLVED.stall_escaped` for P9 instead of
  recomputing. That grades the host's own bookkeeping.
- Fewer than 3 qualifying runs per arm → P9 INCONCLUSIVE (not invalid; see
  result map).

An `--integrity` failure is an abort condition above, and its runs are invalid
for any claim about what governed behaviour.

---

## Result map

| result | next action |
|---|---|
| **P9 complete separation, P10 <= 0.50, P1–P4 hold** | Stall-clocked authority review changes behaviour, and continuation is not a rubber stamp. Retain. Next is the bounded-search card, which has been waiting on this, with its own card and its own contrast. |
| **P9 separation, P10 > 0.50** | Stalls end, but `continue` is doing it by drift rather than by recommitment. The mechanism works for the wrong reason. Next card puts teeth on `continue`: require the falsifier to be host-checkable at authoring time, reject the ones that are not. |
| **P9 no separation, P10 high, dispositions almost all `continue`** | The ritual branch. The review produces well-formed objects and changes nothing. **Delete the typed object.** Retest the interrupt alone — suspend, force one length-1 action, no schema — as q6-del, new card. |
| **P9 no separation, dispositions mixed, escape unchanged** | Stall is a real signal but arbitration is not the remedy. This is the strongest remaining case for the search card, and it becomes next. |
| **P9b collapses in treatment (stalls become rare) with completion flat** | The trigger is being gamed: rationales diversify without behaviour changing. Check the Jaccard measure against action-tuple diversity before believing any escape number. |
| **P9 INCONCLUSIVE (< 3 qualifying runs per arm)** | Do not re-run at a larger budget. Either the chassis stopped stalling, which is itself reportable, or the operating point is too strict — re-derive it from these six runs' traces with `sweep.py` and write a new card. |
| **P1 or P2 collapse** | The review displaced the control loop or the scribe. Restore the frozen chassis and stop adding. |
| **Any Invalid-if** | Do not interpret. New id, new card. Preserve the runs. |

---

## Outputs

`analysis/local-probes/20260905T-stall-authority-q6/results/nemo_solver/<timestamp>_ls20_mdfiles_q6{c,t}<n>/`
— as q4, plus `ipc/authority.jsonl`. Scored by `score_q6.py` in this directory.
Arm assignment recorded in `launch.json` per run.

---

## Pre-launch, required, not part of the treatment

1. **Credit.** Remaining >= $30. Last recorded $0.16. Blocking.
2. **Chassis base.** Implement against
   `../20260904T-host-control-loop-q3/patched-source/after/` plus whatever q4
   applied, exactly as q5's `CHANGES.md` specifies. Do not overwrite the stale
   0158 snapshot as a side effect; it already carries 325 uncommitted lines that
   belong to no pin.
3. **Integrity assertion live before the first paid call.** AUTHORED → STORED →
   RENDERED, failing the run on an unregistered semantic mutation. The check
   exists offline as `replay.py --integrity`; it must run in-process, because
   STORED → RENDERED is where the six q4 mutations enter.
4. **Offline dry run.** Drive the trigger and a scripted `StallResolution`
   through the real host with no model, confirming all four event types appear
   and `score_q6.py` reads them.
5. **Control shadow check.** Confirm `ARC_STALL_REVIEW=0` emits
   `STALL_DETECTED` and no `AUTHORITY_REVIEWED`.
6. Launch pattern unchanged from q4: `daemonize_*.py` double-fork (PDEATHSIG),
   bash `set -a` on `~/projects/arc-agi-3/.env`, `--effort-ladder ''`, model id
   `qwen/qwen3.8-27b`, `ARC_DATA_DIR` = worktree root. Watch
   `ipc/states.jsonl` and `ipc/authority.jsonl`, not `q6*.out`.

---

## Standing decisions this card does not reopen

1. Assist beats refusal at a lifecycle boundary (q3a vs q3c).
2. Check Tycho before designing NOOA lifecycle machinery.
3. RHAE is not a gate at these budgets.
4. Thinking-on remains its own card and its own hatch.
5. A capability does not exist until a live Actor can call it; run
   `chassis-capability-audit/reachability.py` before assuming one.
6. Never let an absolute "looks reasonable" threshold substitute for the
   contrast the experiment measures. `--max-per-100 12` nearly accepted a
   detector whose direction was 0.22x. The primary quantity is always the
   failure/success contrast.
7. A mechanism intended to recover from failure must be clocked by evidence of
   failure, not evidence of success. Completion clocks post-success learning;
   stall clocks remediation. q4's discriminator was clocked on level completion
   and was unfireable on the run it was built for.
8. Prior scorers (`score_q2.py`, `score_q3.py`, `score_q3b.py`, `score_q4.py`)
   are left exactly as they ran.
