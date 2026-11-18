# q3a result — 20260904_011341_ls20_mdfiles_q3a

Card: `../../../EXPERIMENT-CARD.md` (frozen before launch, with a dated pre-launch
amendment block). Scorer: `../../../score_q3.py`. Incumbent: q2d,
`../../../../20260903T-skill-following-q2/results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/`.

Observation, interpretation, and permitted next action are kept separate below.

## Validity

Not invalid.

| check | value |
|---|---|
| `reasoning_tok` | 0 on all 244 calls |
| agent turns | 30 (floor 15) |
| PDEATHSIG | survived; orchestrator reparented to `systemd --user` |
| game id in an emitted cell | none |
| gate exceptions / `observations.json` write failures | none in `launcher.err` |
| H1 live | turn 4: 7 actions, `diff_summary` = 62 cells, last action's own diff = 0. Under the pre-q3a code that turn would have reported 0. |
| H2 live | 6 truncations, each with a note in the state stream |

## Observation — the gate

| clause | q2d | q3a | threshold | |
|---|---:|---:|---:|---|
| P1 search-backed batches | 0.000 (of 16) | **0.889** (8 of 9) | ≥ 0.50 | pass |
| P2 helper refresh on level-up | 0.000 | **0.000** | ≥ 1.00 | **FAIL** |
| P3 re-entry rate | 0.220 | **0.013** | ≤ 0.05 | pass |
| P4 waste share | 0.285 | **0.108** | ≤ 0.15 | pass |

**VERDICT: FAIL** — the card required all four.

Counter-outcome flag also fired: `median_batch = 1.0` and `levels_completed = 1`.

Recorded, not the gate:

| | q2d | q3a |
|---|---:|---:|
| env actions executed | 123 | 74 |
| of those, harness force-advanced | 0 | 5 |
| P4 excluding forced actions | — | 0.101 |
| levels completed | 1 | 1 |
| game_overs | 1 | **0** |
| RHAE game score | 0.030 | 0.011 |
| level-0 actions vs baseline 22 | 24 | 39 |
| retrodiction per multi-action batch | none submitted | 6/7, 6/6, 6/6, 2/2, 3/4, 3/3 |
| rejections | n/a | 2 (1 reentry, 1 stale_encode) |
| output tok / env action | 787 | 2091 |
| wall | 1492 s | 8100 s |
| spend | $0.86 | $2.56 |

## Interpretation

**The control machinery works; it did not buy persistence or depth.**

Three of four clauses passed, each a large move off the incumbent. The actor
predicted the outcome class of its own actions and was right on 8 of 9 multi-action
batches. Waste fell from 28.5% to 10.8%. Re-entry into an already-proven-inert
`(playfield, action)` pair fell from 27 occurrences to one refused attempt — and the
run recorded **zero deaths**, against q2d's one, which was itself a re-entry.

P2 failed, and the failure is specific and useful. At the level boundary the gate
demanded that `encode` run against the new grid, and it got that: the model was
refused once, then supplied `z = {"entity": [35, 29]}`, and the coordinate tracked
correctly on subsequent turns. But the helper module was written only at env steps 0
and 3 and never again. **The refusal forced the computation and not its persistence.**
The model satisfied the gate inline, in a cell, and nothing survived the turn.

The counter-outcome flag fired but its attached diagnosis — "compliance without
competence" — is falsified by P1 = 0.889 and P4 = 0.108. `median_batch` is the wrong
statistic here: the policy is bimodal, not uniformly cautious. Nine batches were
longer than 1 (including 7, 6, 6, 4, 3, and a 19-action burst in the last two turns),
interleaved with single-action probes. A median cannot see that shape. This is a
defect in a metric I wrote, noticed after seeing the result, and it is recorded as
such rather than used to reinterpret the branch quietly.

On score the run is **worse than the incumbent**: RHAE 0.011 vs 0.030, because level 0
took 39 actions against q2d's 24 (baseline 22), and because the completing action at
step 39 was a harness force-advance rather than a chosen move. The card forbids score
claims from this run, and that prohibition now cuts against the treatment, not for it.

**The dominant failure was operational, not conceptual.** Five force-advances and six
urgent nudges. The actor ran 8–18 LLM rounds per turn at up to ~80,000 prompt tokens,
overran the 300 s turn timeout, and the harness played a default action for it 5 times
in 74. Output per env action tripled (787 → 2091) and wall time 5.4×. The treatment
asked for apparatus and got it, along with cell output that re-accumulates every
round. `--agent-turn-timeout 300` was calibrated for the terser actor q2d was.

## Permitted next action

1. **Retain the host control loop.** P1, P3, P4 all passed with margin; zero deaths.
   Do not revert truncation, the observation record, `predicted_path`, or the
   re-entry gate.
2. **Fix P2 with a scribe, not a stronger refusal.** Tycho already implements this
   and q3a did not run Tycho. `tycho/agent/agent.py:652` registers
   `AgentSpec(id="scribe", trigger=on_events(Event.LEVEL_COMPLETE), kind=AgentKind.LLM,
   sees_full_history=True, run=_summarize_level)`; `_summarize_level` consolidates
   terminal evidence before the level transcript resets and writes it — to
   `world_model.py` in single mode, to `notes/world_model.md` in builder-owned modes —
   with a `_write_level_insight_fallback` when the call budget is exhausted. A companion
   trigger-mode builder fires on `LEVEL_START` for the same reason.

   The NOOA solver q3a runs has no scribe, no agent graph, and no `LEVEL_COMPLETE`
   event, so the boundary carried only gate G2. G2 can make omission illegal; it cannot
   perform the consolidation. That is the refusal-versus-assist distinction this line of
   work already recorded, and the earlier version of this recommendation — "require a
   persisted artifact at the gate" — reinvented the weaker half of a mechanism that
   exists and works. `Tycho-NOOA Architecture - What Remains to Steal.md:323` lists
   "scribe/summarization only at meaningful lifecycle boundaries" among the things to
   take; it should be ported to the NOOA solver rather than re-derived.
3. **Raise `--agent-turn-timeout` (600+) and keep the escalation ladder.** The URGENT
   nudge demonstrably broke an over-deliberation loop at t30. Both, not either.
4. **The counter-outcome branch is not licensed here.** Its trigger fired on a
   statistic that cannot describe a bimodal policy. Do not read this run as a case
   for reasoning-on. The thinking-on card remains open and unaddressed by q3a.
5. **No score claim.** A run that speaks to depth needs a larger turn budget and its
   own card; at 40 turns and 120 env steps the reachable RHAE is capped near 0.10
   regardless of actor quality.

## Scorer amendments (all before any q3a result existed, except where noted)

1. P3 counted over attempts rather than executions, since the host now refuses
   re-entries at submit time — pre-launch, documented in the card.
2. Per-action frame alignment rebuilt on `states.jsonl` `action_results` instead of
   `actions.jsonl` batches. Force-advances execute env steps with no batch entry and
   desynced the frame index. Correctness bug, found mid-run, no threshold touched.
3. P2 rewritten to use env-step invocation ids; the earlier version compared message
   *round* numbers against agent *turn* numbers. Correctness bug, found mid-run.
4. `P4_waste_share_model_only` added as a companion. The card's all-actions
   definition remains the gate. Added mid-run, before any P4 value was computed.
