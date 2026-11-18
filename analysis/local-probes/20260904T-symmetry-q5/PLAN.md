# q5 plan — give the actor a comparison vocabulary

Written 2026-09-04 while q4 is live (PID 424197, tag `q4`, launched 16:50).
**This is a plan, not a frozen card.** The card cannot be frozen until q4 scores,
because q4's outcome selects q5's contrast (§6). Predecessors:
`../20260904T-goal-candidate-q4/EXPERIMENT-CARD.md`,
`../20260904T-host-control-loop-q3/` (q3a/b/c), `../20260903T-skill-following-q2/`.
Origin: `../../chassis-capability-audit/README.md`.

> **Revised 2026-09-04 after an oracle disclosure.** Pranav supplied ls20's actual
> mechanic (`ORACLE-ls20.md`, quarantined). It answers §3's applicability question in the
> affirmative and sharpens §1, §8. §3 is kept — its purpose changed from go/no-go to
> scoring key. Read `ORACLE-ls20.md` before this file; obey its handling rule.

## 1. Question

q3c had 19/19 dynamics on an unseen level and a wrong goal predicate it defended for
eight turns. **Every level-1 `predicted_path` expect was `move`.** q4 supplies a record
that demands a terminal-class discriminator; it supplies no means of computing one.

The mechanism is now established, not inferred. q3c's `Z_SCHEMA` was
`{grid, mover, mover_size, items}` plus a hardcoded level-0 goal rectangle. The two
variables that decide ls20 — the key glyph and the door slots, both plainly visible in the
public grid and both changing discretely — **are absent from the latent state**. The model
was perfect on what it encoded and blind to what it did not. Dynamics could be 19/19 while
the objective stayed wrong, because the objective was not representable.

So the question is not "will it call `symmetry`". It is:

**Does a comparison vocabulary get the goal-carrying state into `encode`, and out of
`encode` into a terminal-class discriminator?**

## 2. Why symmetry and not the planner

The audit found both orphaned. Symmetry is the better q5:

- It targets frontier 4 (objective discrimination), the recurrent weak link named in
  board 119 and re-confirmed by q3c.
- It has a prior positive with a legible mechanism (uptake `20260820T101248Z`,
  replicated `141506Z`): docstring read → quotient chosen → `congruent` True →
  14-move transport costed → level completed.
- The planner's benefit on `ls20` is unrealized-risk reduction. q3c's unbounded
  hand-written BFS *worked* (P1 = 0.944) because the level-1 reachable set is 5×5.
  A run there would have almost no discriminating power. The planner belongs on a game
  with a large state space, or after a hang recurs.

## 3. Step 0 — the offline applicability probe (do this first; it costs nothing)

~~**The strongest objection to q5 is that `ls20` may not be a game where congruence
decides anything.**~~ **Answered, 2026-09-04.** ls20's win condition *is* an equivalence
statement — key glyph matches lock glyph — and both figures are in the public channel,
changing discretely (`ORACLE-ls20.md`). Congruence is not merely applicable here; it is
the win condition's own form.

The probe below is therefore no longer a go/no-go. It becomes the **oracle scoring key**:
the relation is computed by us, in advance, so we can tell whether the actor found it.
It is quarantined and does not ship.

q3c left 203 full grids on disk
(`../20260904T-host-control-loop-q3/results/nemo_solver/20260904_140403_ls20_mdfiles_q3c/steps/step_*.json`,
each with `grid`, `level`, `level_completed`, `event`) plus `ipc/level_0_terminal.json`.
That is enough to answer the question for zero tokens, in the shape of the
`no-impact-offline/` probe already accepted in q2.

Probe: build figures from connected components of each frame; for the observed level-0
terminal and its pre-win frame, test whether any of
`canon` equality / `transporter` non-empty / `congruent` / `primitive` between the mover
and any candidate target region **separates the terminal frame from the 200 non-terminal
frames**. Write the separating relation down, or write down that none exists.

| probe outcome | consequence |
|---|---|
| a relation separates the terminal | q5 proceeds on `ls20`; the relation is the existence proof, and is **not** given to the actor |
| no relation separates it | q5 does **not** run on `ls20`. Either move the game, or q5 becomes the planner port. Record the negative — it is a real finding about which games symmetry can serve |

The separating relation must never enter the shipped brief or the run. It is our
evidence that the question is answerable, not the actor's answer.

## 4. Feasibility, established rather than assumed (2026-09-04)

- `symmetry_template.py` is pure Python, zero imports, 10,566 bytes.
- Lifting the live guard out of the q3 chassis and running it:
  `_check_helper_ast(symmetry_template) -> None`. It passes unmodified. No allowlist edit
  is needed, so the L1 isolation guarantee is untouched.
- It defines neither `encode` nor `predict`, so `load_helpers()` cannot mistake it for
  the world model (`_active_world_model_name` is unaffected).
- `load_helpers()` re-checks the AST on load precisely because "a seeded helper could
  predate the write-time guard" — seeding is an anticipated path, not a hack.
- A calibrated brief already exists: `tycho_nooa/runtime_notice.py:26` `_SYMMETRY_TOOL_NOTE`.
  It was rewritten after the uptake run to name every admissible parameter value, which is
  the fix for the measured defect in §5.

## 5. Treatment — one factor

Seed `helpers/symmetry.py` (byte-exact `symmetry_template.py`) into the run workspace
before launch, and surface it. Nothing else changes.

Surfacing follows the one design fact this codebase has actually settled — **assist, not
refusal** (q3a's gate: P2 = 0.000; q3c's evidence + status line: P2 = 1.000, zero
rejections):

- the status line gains a `provided:` entry naming the module's public functions and one
  pointer sentence (CHANGES C2/C4).

**Superseded:** an earlier draft put `_SYMMETRY_TOOL_NOTE` in the cell tool description.
CHANGES C4 is the law instead — the teaching text ships inside the module's own docstring,
so it travels with the bytes and an unset seed cannot advertise a missing tool. Do not
resurrect the prompt-side brief.

No new refusal, no gate on using it, no skill rewrite.

**Why verbatim.** The one law this executor has demonstrated twice is that it uses the
vocabulary it is handed, literally, and does not generalise a parameter's type from its
neighbour: `canon(point='full')` failed 0/2 when `point` took an object and `colours` took
a string, and `rationale` scored 0/24 when the chassis said `"prediction..."` and the skill
said `predict:`. Rewriting a brief that was already repaired against the first of those is
how we re-earn it.

## 6. Contrast — branches on q4

| q4 result | q5 contrast | note |
|---|---|---|
| PASS | q4 + symmetry, one factor | clean; q4's P1–P4, P6, P7 all become hold-the-line |
| FAIL on P6 (record produces no executed discriminator) | q4 + symmetry, still one factor | the more interesting case: q4 shows the record alone is not enough, and q5 asks whether the missing piece was the *means* |
| FAIL on P1–P4 (frontier-4 ritual displaced the control loop) | **q3c + symmetry**, q4's record excluded | do not stack a factor on a treatment that broke the loop |
| INVALID | hold; re-run q4 first | |
| **INCONCLUSIVE — what happened** | **q3c + seed. q4's record absent, or present and recorded-not-gated.** | see below |

**q4 result, 2026-09-04:** `max_turns` at 60, 197 env steps, `levels_completed = 0`,
`ipc/goal_candidate.json` never written, $3.56. The record never activated, so there is
nothing to stack on. Stacking anyway is two untested treatments on one boundary: if a
level completes this time, P6/P7 and S1/S2 all fire at once and neither is attributable;
if it again never completes L0, holding P6/P7 as gates just reprints INCONCLUSIVE.

**Contrast is q3c**, the last run that completed a level and has scored gates. q4's P6/P7
may ride along as recorded-not-gated. q4's own result map said stop that line, not raise
the budget and add a tool.

Held fixed in every branch: game `ls20`, variant `mdfiles`, model `qwen/qwen3.8-27b`,
skill `interactive-game-solver`, `--visual off`, `--effort-ladder ''`, thinking off via
`ARC_LLM_EXTRA_BODY`, sandbox `inproc --sandbox-cells --sandbox-require --sandbox-cpu-s 60`,
`--agent-turn-timeout 600`, `--max-turns 60`, `--max-env-steps 200`, daemonized launch.

## 7. Unit of analysis — fixed now, before any data

q2d's gate was scored FAIL and PASS from the same run because the denominator was chosen
after the fact (env steps vs turns). Not again. For q5:

- **agent turn** = one `agent_turn` event in `events.jsonl`, the same unit `result.json`
  reports as `turns`.
- **executed symmetry reach** = a `repl_execute` cell whose source matches
  `\bsymmetry\s*\.\s*(\w+)` **and** whose trace row has `status == "complete"` with no
  `Traceback (most recent call last)` in its result. This is `tools/symmetry_uptake.py`'s
  rule, ported to the nemo trace format; a reach that raises is a failed reach, never an
  uptake success.
- **level-transfer** = a `levels_completed` increase followed by ≥1 further agent turn.
- **discriminator** = as `score_q4.py` defines it, unchanged.

## 8. Gate

| clause | measure | prior | threshold |
|---|---|---:|---|
| **S1** (necessary) | executed symmetry reaches | 0 (unreachable in every NOOA run) | **≥ 1** |
| **S2** (the question) | ≥1 executed discriminator whose `predicted_path` carries a `level_up`/`terminal` expect **and** whose turn contains an executed symmetry reach | 0 | **≥ 1** |
| **S3** (recorded, not gated) | `Z_SCHEMA` acquires a field derived from a figure comparison — i.e. the goal-carrying state enters the latent | 0 (q3c: `{grid, mover, mover_size, items}`) | report |
| **H1–H4** | q4's P1–P4 | q4's values | no regression below q4's thresholds |

S2 is binary and low-n by construction — a run reaches one or two level-transfers. It is
stated as a binary observation, not a rate, and it is **not** the kind of number to
average across runs later.

Null: the actor never executes a symmetry call, or executes some and never uses one to
state a terminal-class prediction. Both are real outcomes with real consequences (§9).

## 9. Result map

- **S1 ≥ 1, S2 ≥ 1** → the vocabulary converted a goal hypothesis into a test, in the
  NOOA chassis. Next is the **sham arm**, not a victory lap: a module of the same brief
  length with non-geometric functions, to separate "symmetry" from "another tool appeared
  in the status line". Board 161/163 already require sham arms for exactly this.
- **S1 ≥ 1, S2 = 0** → reached for, but not for the objective. This is the uptake run's
  own middle branch repeating in a new chassis. It argues the constraint is that nothing
  *asks* for a geometric terminal signature, which is a host question, not a tool question.
- **S1 = 0** → the status line and tool note are not enough to make a seeded capability
  live. That is a discoverability finding about the chassis and it invalidates the surfacing
  design, not symmetry.
- **H\* regression** → symmetry displaced the control loop; revert and report.

## 10. Caps, and the budget wall

**q5 cannot be paid for. Remaining credit is $2.75 against a $6 floor (2026-09-04, after
q4 spent $3.56).** The card does not get written until remaining ≥ $6. Below is the
reasoning from before that number was known.

**q5 probably cannot be paid for yet.** `q4-spend-start.json` records
`remaining: 6.31854497` against `cap: 6.0`. If q4 spends its cap there is ~$0.30 left, and
q3c cost $3.68 for 38 turns / 202 env actions. **Confirm remaining credit ≥ $6 before
writing the card**; if it is not there, q5 waits on a top-up.

This is why §3 is step 0 and not an appendix: the applicability probe is free, runs today
against artifacts already on disk, and can kill or redirect q5 before a dollar is at risk.

Other caps, when it does run: 60 agent turns, 200 env steps, 4 h wall, $6, abort on
transport failure or mid-run cap breach. Scorer `score_q5.py` written and frozen before
launch; `score_q4.py` left exactly as it scored q4 and not reused.

## 11. Contestant prior

`CONTEXT.md` admits into a Contestant Prior "the basic ARC-AGI intelligence priors needed
to observe, act, remember, **compare**, learn, and adapt". `symmetry` is game-general
lattice geometry: it encodes no object roles, coordinates, visual conventions, goals,
resources, or transition rules, and its own docstring refuses to supply a win condition —
*"There is no ready-made helper for YOUR game's win condition, and there will not be."*
Admissible as shipped. The §3 separating relation is **not** admissible and does not ship.

## 12. Claim ceiling

One game, one seed, one model, one run, and — if S2 fires — one binary event. It would
establish that this executor can be handed a comparison vocabulary and use it for the
objective in this chassis. It would establish nothing about 25 games, nothing about the
hidden half, and nothing about RHAE. Symmetry's own prior is n=1 plus one
same-configuration replication, on a different game, in a different chassis.

## 13. Not in this plan

- The planner port (`plan_bfs` and friends) — the next audit item, its own card.
- Any change to q4. Keep q4 one-factor; do not fold symmetry into a running experiment.
- WikiSkill / evolved `SKILL.md` — board 267's deferral stands.
- Segmentation, dynamics probes, death indexing — audited as absent, unscheduled.
- Thinking-on. Still its own card.

---

## 14. Two findings from the q3c frames, added 2026-09-04

Both recomputed from `steps/step_*.json`. Neither is folded into q5.

### 14.1 `encode` was blind to the goal state — this is the real q5 target

q3c's `Z_SCHEMA` is `{grid, mover, mover_size, items}` with `GOAL` a hardcoded level-0
rectangle. The key glyph (rows 54–60, cols 0–11, changing at 8 steps) and the door slots
(rows 61–62, cols 55–63, flipping at 5 steps) never enter the latent. Perfect dynamics over
an incomplete state is exactly what 19/19-plus-wrong-goal looks like. Recorded as S3.

### 14.2 H2's no-impact rule suppresses one real door event — flagged, not fixed here

H3 discovers the no-impact band at **row** granularity: rows 61–62 change on 97.5% of steps
and are the only rows above the 0.8 threshold. But those same two rows carry the door slots,
which change discretely. Consequences on the q3c trajectory:

- 13 steps would be truncated as no-impact; 4 have genuinely empty diffs.
- **Step 141 is a door-slot flip whose entire diff lies in rows 61–62.** H2 would call the
  door event a no-op and drop the rest of the batch. 1 of 5 door events, not all — the
  other 4 also move playfield cells and survive.
- No key change is affected: the key sits at rows 54–60, outside the band.

**A repair I tried and rejected.** Discovering the band at *cell* granularity at the same
0.8 threshold yields an empty band — the clock erodes one cell at a time, so no single cell
clears 0.8 — collapsing H2 to "truncate only on an empty diff" and losing HUD suppression
entirely (13 truncations → 4). The naive fix over-corrects. A correct rule has to separate
a cell carrying discrete state from a cell inside a sweeping band, and that needs design.

Its own card, after q4 and q5. Do not patch it into a running or planned experiment.

## 15. Open decision for Pranav — does the brief name the *form* of goals?

The oracle claim is that many ARC-AGI-3 games are "act until X matches Y / fits in Z /
matches level". If true, that is a statement about the **task family**, not about a game,
and `CONTEXT.md` admits family-level priors ("observe, act, remember, **compare**, learn,
adapt") into a Contestant Prior.

- **Ships today:** the `symmetry` module and its brief, which teach *how to compare* and
  explicitly refuse to supply a win condition.
- **The question:** may the brief add one sentence — *"in these games the objective is
  often an equivalence: act until one thing matches another"* — which teaches *where to
  point* the comparison?

My recommendation: **not in q5.** It is plausibly admissible, but it would make q5 a
two-factor run (tool + goal-form hint) and the tool is the cheaper thing to establish
first. If S1 fires and S2 does not, that sentence is the obvious q6 arm and the result map
already predicts it. Pranav's call, since it is a competition-legality judgment, not a
technical one.

`symmetry` also does not cover "fits inside": its API is equality-under-a-group, plus
`primitive` for scale. Containment or embedding would be a genuine extension, not a preset.


---

## 16. Status, 2026-09-04 — BLOCKED, two independent reasons

1. **The offline separator probe came back negative** (`PROBE-RESULT.md`). PLAN §3's own
   rule then binds: do not freeze the card against ls20.
2. **Remaining credit $2.75 against a $6 floor.**

Both are cheap to resolve in the right order: keep probing offline (free) before asking
for budget. Findings that survive regardless: `CHANGES.md` C1–C4 and `reach_scorer.py`,
both validated and neither ls20-specific.
