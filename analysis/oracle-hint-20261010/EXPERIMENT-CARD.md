# Oracle-hint experiment (2026-10-10)

## Question
On the levels our incumbent fails, is the binding deficit **goal judgment** (it would solve the level if told the goal),
**execution** (it needs the route too), or **perception/interface** (it fails even with the route)?
Context: memory `harness-paradox-resolution`; Makarov 743723; `metacog-transfer-20261008`.

## Units
40 hard level entries from the three dfranzen runs (control, r2, s1): **26** levels stuck in ≥1 run and solved by a sibling,
**14** reached and never solved. `starts.json` (built by `starts.py`): each prefix lands on level L on the raw engine,
every run reaching L lands on the identical frame, and every sibling's solving segment solves from our prefix (26/26).

## Arms (one 1,200 s sprint per attempt, fresh context at level entry, unchanged control model/serving/harness)
| arm | system-prompt addition | replicates |
|---|---|---|
| A0 | none | 2 (session `a0`; the two replicates are also the A/A noise check) |
| A1 goal | the level's win condition in board terms; no coordinates, no action names, no sequence | 1 (session `hints`) |
| A2 plan | A1 + what the controls do + ordered subgoals; no coordinate list or literal action sequence | 1 (session `hints`) |

Only differences from the control notebook: plain 10-way pool (priority scheduling off), 1,200 s per attempt,
attempts start at level entry (replayed prefix), an attempt ends when its level is completed, the hint line.
All three apply equally to every arm except the hint. Sessions are separate Kaggle runs on the same image and config;
attempt order is shuffled (seed 20261010) within each session.

Hints are written from the game source and, for sibling levels, the solving run's trace, before any A0 result is read.
Each plan is checked against an action sequence that solves the level on the engine. A blind review checks that goal
hints don't leak actions. Hints that can't be verified are flagged, and that level is reported separately.

## Outcome
solve = the level is completed within 1,200 s of admission (`events.jsonl` kind=solved, or benchmark state `won`).
Per level: p0 = mean of the two A0 replicates; d1 = A1 − p0, d2 = A2 − p0. Report means over the 40 levels with a
level-bootstrap 90% interval, and separately for sibling (26) and never (14) levels.

## Decision rule (pre-registered)
- mean d1 ≥ 0.30 → **judgment**: the gap is choosing the goal; harness work on execution has a low ceiling here.
- mean d1 < 0.15 and mean d2 ≥ 0.30 → **execution**: closure/route tools (simulators, search, retention) are the lever.
- mean d2 < 0.15 → **perception/interface**: the model cannot act even when told; inspect representation.
- otherwise → inconclusive; a second hint replicate is the next spend.
Secondary (descriptive only): A0 r1 vs r2 agreement; time-to-solve among solves; transcripts of A1 failures
(ignored hint / misread board / failed execution).

## Budget
Session `a0`: 80 attempts, hard cap 4 h. Session `hints`: 80 attempts, hard cap 4 h. Experiment ceiling 8 GPU-h of the
30 h week (≈5.1 h used by B1 before launch). One push attempt per session; a launch-attempt.json means never push again.

## Known limits
Fresh context drops cross-level learning, so absolute rates sit below in-run rates; the contrast is paired and unaffected.
A judgment verdict locates the gap but does not supply a lever (fine-tuning is out: `solo-no-finetune-bundle-plan`).
The hints are frontier-written, so A1 measures frontier goal judgment transplanted into our stack.

## Amendment 1 (2026-10-10 ~18:40Z; session a0 running, no A0 output read)
1. **A2 is goal + literal verified route.** The plan arm gives the A1 goal text plus a route (the exact action list, in the
   model's own action() vocabulary) that `routes.py check` verifies on the engine from the level entry. Hand-written
   object-level plans proved too costly for levels with reshaping or long setup (cn04 L5), and a literal route is the
   cleaner ceiling: A2 − A0 small means the model fails even when told exactly what to do (interface/perception).
2. **The hint session covers levels with written hints, sibling levels first.** Hints for the 14 never-solved levels need
   the win condition traced through obfuscated game source, plus a route found by search. They join only if both
   are done before launch. The primary analysis is on the sibling levels that have hints; never-solved levels without
   hints keep their A0 baseline (descriptive).
3. **A1 gets two replicates** (A1 r1, r2; A2 one). Session `hints` = 3 attempts per hinted level. That's ≤ 78 attempts,
   inside the same 4 h cap. p1 = mean of the two A1 replicates; d1 = p1 − p0.

## Note 2 (2026-10-10 ~17:55Z, before any output): fresh-context confound
In-run, a stuck level is reached with whatever earlier levels of the same game taught the model. A0 sprints start cold,
and A1 hints restore part of that game-level knowledge as well as the level-specific goal. A1 − A0 is therefore an upper
bound on the in-run judgment gap. The readout reports each level's in-run solve fraction (3 dfranzen runs) next to p0.
If p0 is far below the in-run rate, part of d1 is restored context rather than judgment.
