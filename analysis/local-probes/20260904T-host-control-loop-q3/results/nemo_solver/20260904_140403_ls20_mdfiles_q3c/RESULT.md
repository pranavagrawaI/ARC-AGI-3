# q3c result — 20260904_140403_ls20_mdfiles_q3c

Card: `../../../EXPERIMENT-CARD-q3c.md`. Scorer: `../../../score_q3b.py` (unchanged,
reused). Incumbents: q3b (same dir), q3a, q2d.

## Validity

Not invalid. `reasoning_tok = 0` on all 308 calls. 38 agent turns (floor 15). One level
boundary reached and `ipc/level_0_terminal.json` written. No game id in an emitted cell.
Orchestrator survived (reparented to `systemd --user`). Terminated on `max_env_steps`
(202 of 200).

## Gate

| clause | q2d | q3a | q3b | **q3c** | threshold | |
|---|---:|---:|---:|---:|---:|---|
| P1 search-backed batches | 0.000 | 0.889 | 0.875 | **0.944** | ≥0.50 | pass |
| P2 model-written consolidation | — | 0.000 | INCONCLUSIVE | **1.000** | ≥1.00 | pass |
| P3 re-entry rate | 0.220 | 0.013 | 0.011 | **0.020** | ≤0.05 | pass |
| P4 waste share | 0.285 | 0.108 | 0.057 | **0.064** | ≤0.15 | pass |

**VERDICT: PASS.**

P5 (recorded, not gated): **0.0** — the helper was not rewritten after the boundary.

## Observation — the scribe fired end to end

1. Level 0 completed in **32 actions** (baseline 22; q3a 39; q3b never).
2. The harness wrote `level_0_terminal.json` — winning action `UP`, pre-win frame,
   terminal frame, 9,363 bytes.
3. The model **called `self.level_terminal()`** and compared the three.
4. It wrote `level_0_insights.md` (1,378 B) itself. **Zero scribe rejections** — the
   `SCRIBE PENDING` status line sufficed; the gate never had to refuse anything.

The note contains a goal predicate, not a summary:

> *Items (0/1 cross) = KEY to unlock box entry*
> *Tile can overlap box border (3s) but can't fully enter box WITHOUT items*

plus a `## What Generalizes` section: *"Items unlock areas. Goal is in a restricted
area requiring items."*

**The refusal-versus-assist question is settled for this codebase.** q3a's gate demanded
the artifact and got nothing (P2 = 0.000). q3c handed over the evidence and got the
artifact without ever refusing a submit (P2 = 1.000). Same actor, same game, same model.

## The predicate transferred, and then misled

Post-boundary rationales carry it onto level 1 — *"Navigate to items at (45,49) to
unblock goal"*, *"Navigate to goal at (40,14) after picking up items"* — against q3b's
*"Testing untried position"*. Batches grew to 13/17/11/19/20/17: searched routes, not
probes. Ledger H4 (*"correct artifacts are often not causally used"*) does **not** hold
here; the artifact was used.

Its level-1 dynamics were near-perfect: retrodiction **19/19, 16/16, 19/19, 16/17,
10/11**.

And it still did not clear level 1, because on that level the goal sits inside a 5-cell
enclosure and the transferred predicate is wrong. Eight turns went into re-testing the
level-0 theory — proximity, item pickup, DOWN entry, passing through 9-cells — before
t28 reached *"maybe the win condition is not entering the 5-cell box"*.

| | |
|---|---|
| transition dynamics on an unseen level | 19/19 |
| goal predicate | transferred, wrong, held for eight turns |

The scribe produces a predicate. **Nothing requires that predicate to be
re-discriminated on the next level.** It arrives as an assertion rather than a
hypothesis carrying a cheapest discriminator and a deadline in actions, so a correct-but-
stale objective was defended by flawless execution. That is ledger Q2, and this run is
the strongest evidence for it we have.

## Economy

| | q2d | q3a | q3b | q3c |
|---|---:|---:|---:|---:|
| calls | 129 | 244 | 341 | 308 |
| env actions | 122 | 74 | 88 | **202** |
| LLM calls / action | 1.06 | 3.30 | 3.88 | **1.52** |
| actions / submission | 5.08 | 2.47 | 2.38 | **5.77** |
| output tok / action | 787 | 2091 | 2049 | **903** |
| input tok / action | ~57k | — | 217k | **95k** |
| force-advances | 0 | 5 | 1 | **1** |
| spend | $0.86 | $2.56 | $3.13 | **$3.68** |

Action economy recovered to roughly q2d's level *with* the full control loop attached —
5.77 actions per submission at 1.52 calls per action. That came from the consolidation
producing routes worth batching, not from any tempo change.

**The context budget did not work.** Mean prompt tokens: q3a 56,359, q3b 55,914,
**q3c 55,883**, and `<truncated>` appears in zero cells in any run. `capture.max_stdout`
was never applied. Most likely cause: `max_context_tokens` was set while
`max_event_tokens` was left `None`, and the accumulated payload here is event history.
Every tempo claim in the q3c card is void; the per-action improvements above come from
larger batches, not smaller prompts.

RHAE 0.0169. No score claim is drawn: 200 env steps caps the reachable game score near
0.10 regardless of actor quality.

## Permitted next action

1. **Retain the scribe.** Port Tycho's `LEVEL_START` trigger-builder next — the
   companion mechanism for the same boundary.
2. **The next card is frontier 4, not dynamics.** A goal-candidate record carrying a
   host-scored terminal signature, a cheapest discriminator, and a **deadline in
   actions**, so a transferred predicate must be re-discriminated on a new level instead
   of defended. This run supplies the evidence; ledger Q2 supplies the specification.
3. **Fix the context budget properly**, and verify with the `<truncated>` marker
   *before* spending: set `max_event_tokens`, confirm markers appear in cells, then run.
4. **P5 stays open.** The predicate lives in markdown and drives reasoning directly; it
   never reached `predict`. Milder than q2d's failure but the same shape.
5. No score claim from this budget.
