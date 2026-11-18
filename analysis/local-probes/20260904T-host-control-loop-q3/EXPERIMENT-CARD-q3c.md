# q3c-scribe-with-context-budget

Written 2026-09-04 after q3b scored INCONCLUSIVE. Frozen at launch.
Predecessors: `EXPERIMENT-CARD.md` (q3a), `EXPERIMENT-CARD-q3b.md`, and both RESULTs.

Question: unchanged from q3b, because q3b never answered it — does the ported
`LEVEL_COMPLETE` scribe produce a **model-written** carry-forward at a level boundary?
q3b ran 40 turns and 88 actions without completing a level, so the mechanism never
fired. This run buys enough budget to reach a boundary, and makes the run cheap enough
that the budget is affordable.

Posture: repeat of an inconclusive measurement, with the two constraints that made it
inconclusive removed.

Contrast: q3b, `results/nemo_solver/20260904_035318_ls20_mdfiles_q3b/`.
Held fixed: game, variant, model, skill, `--visual off`, `--effort-ladder ''`, thinking
off, sandbox settings, `--agent-turn-timeout 600`, every gate including the scribe.

Changed, two things:
1. **A context budget** (below). q3b's framework defaults left the base context
   unbounded.
2. **`--max-env-steps` 120 → 200, `--max-turns` 40 → 60.** q3b ran out of turns seven
   steps into a ten-step plan toward its first well-founded goal hypothesis.

Neither change touches what P2 measures. Both change tempo, so tempo comparisons
against q3a/q3b are directional only.

---

## The context budget, and why these numbers

Measured on q3b: 341 LLM calls, **19,066,945 input tokens**, mean 55,914 per call,
180,341 output tokens. 100% of the 88-minute wall was LLM round-trips; env steps cost
~80 ms total. Latency tracks prefill — calls under 30k tokens ran 7 s median, calls at
or above 60k ran 12 s.

The accumulation is **across turns, not within them**: marginal prompt growth per round
summed to only 431,804 tokens over the whole run. The 19M comes from paying a ~56k base
context 341 times. Three framework defaults allowed that, none of which the ARC solver
had ever set:

| knob | default | q3c | why |
|---|---|---|---|
| `max_context_tokens` | `None` | **32_000** | with no budget, L4 eviction never runs and the base grows monotonically. `min_preserved_events=5` keeps recent turns. |
| `capture.max_stdout` | 50_000 chars/cell | **6_000** | a printed 64×64 hex grid is ~4,400 chars, so one still fits; repeated dumps are clipped. This enforces the skill's own "compute, don't squint". |
| `event_format` | `max_string=10_000, max_length=200, max_depth=5` | **2_000 / 100 / 4** | the config's own comment: renders *every turn for the rest of the run*. Highest leverage single number. |

Wired at `launcher.py` where the agent is constructed. **This does not remove
information from the model**: NOOA's contract is that the live variable is untruncated
and re-indexable — only its rendered form is capped. It also cannot impair the scribe,
which reads its evidence from a fresh `self.level_terminal()` call rather than history.

Projected from the q3b trace: a 32k context cap alone puts input at ~52% of actual.
With the observed round distribution that is roughly 9–10M input tokens and a wall
near 40–50 minutes for 60 turns, against q3b's 88 minutes for 40.

## Prediction

| # | metric | q3b | q3c predicted |
|---|---|---:|---|
| P1 | batches >1 whose `predicted_path` scored ≥0.6 | 0.875 | **≥ 0.50** (hold) |
| P2 | completed levels with a **model-written** consolidation | INCONCLUSIVE | **= 1.00** |
| P3 | re-entry rate over attempted actions | 0.011 | **≤ 0.05** (hold) |
| P4 | waste share of executed env actions | 0.057 | **≤ 0.15** (hold) |

P1 is the clause at risk from this change: if a 32k budget evicts something the actor
needed, prediction accuracy is where it shows. A P1 collapse with P2 met would mean the
budget was cut too far, not that the scribe failed.

Recorded, not gated: P5 helper refresh; **input tokens per env action** (q3b: 216,670);
**LLM calls per env action** (q3b: 3.88; q2d, ungated: 1.06); force-advances (q3b: 1;
q3a: 5); wall; spend; actions per submission (q3b: 2.38; q2d: 5.08).

## Measure

`score_q3b.py`, unchanged and reused — it is already frozen against these four clauses
and its verdict logic now reports INCONCLUSIVE rather than PASS for an unmeasured gate.
P2 marker: a note containing `Deterministic fallback note:` was written by the host and
counts as a P2 failure, not a save.

RHAE is not the gate. Even at 200 env steps no score claim may be drawn.

## Cap

60 agent turns, 200 env steps, `--agent-turn-timeout 600`, 4 h wall, **$8** OpenRouter
(q3b: $3.13 for 40 turns; this run has 50% more turns at roughly half the input cost).
Pre-launch: confirm remaining credit ≥ $8.

## Stop

- Score after ≥15 agent turns.
- Abort on spend ≥ $8, wall ≥ 4 h, `--max-env-steps` reached.
- Refusal deadlock: ≥8 consecutive rejected submits (`ipc/ABORT`, enforced in code).
- Abort if `reasoning_tokens > 0`.

## Invalid if

- `reasoning_tokens` return; PDEATHSIG kills the orchestrator; fewer than 15 agent
  turns; game id in an emitted cell; a gate throws and is silently bypassed.
- A level completes and no `ipc/level_<L>_terminal.json` is written — the port did not
  fire and P2 measures nothing.
- **No level completed again** — P2 INCONCLUSIVE for the second time. See below: that
  outcome ends this line, it does not license a q3d.

## Result map

| result | next action |
|---|---|
| **P2 = 1.00, P1/P3/P4 hold** | The port works and q3a's refusal did not. Retain the scribe; the refusal-vs-assist distinction is settled for this codebase. Next: port Tycho's `LEVEL_START` trigger-builder, and a depth run with its own card. |
| **P2 = 0 via host fallback** | The host consolidated and the model would not read the boundary. Keep the fallback; this is the strongest available case for the thinking-on card, which then becomes next. |
| **P1 collapses, P2 met** | The context budget was cut too far. Re-run at `max_context_tokens=48_000` before drawing any conclusion about the scribe. |
| **P2 INCONCLUSIVE again (no level)** | **Stop running this experiment.** Three runs, ~$9, and the actor cannot reach a boundary on `ls20` reliably enough to measure a boundary mechanism. The question then is not "does the scribe work" but "why can this actor not find a goal", which is the q3b finding: perfect dynamics prediction with no goal discovery. That is a different card. |
| **Any Invalid-if** | Do not interpret. New id, new card. Preserve the run. |

## Outputs

`results/nemo_solver/<timestamp>_ls20_mdfiles_q3c/` — as q3b, plus
`ipc/level_*_terminal.json`. Scored by `score_q3b.py`.

## Known, and deliberately not fixed here

q3b's helper carried `walls = (g == 3)` while its own confirmed notes said `3 = walkable
floor, 4 = wall` — an inverted wall map that made its `bfs` search a mirror of the real
map for the whole run. The turn-37 correction reached markdown and never reached the
code. That is a real defect and a candidate for a host check (flag a confirmed note that
contradicts a helper constant), but adding it here would confound P2 a third time. It
gets its own card if q3c resolves the scribe question.
