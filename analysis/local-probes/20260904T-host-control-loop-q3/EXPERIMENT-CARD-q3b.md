# q3b-scribe-port

Written 2026-09-04 after q3a scored and before q3b launched. Frozen at launch.
Predecessor: `EXPERIMENT-CARD.md` (q3a) and that run's `RESULT.md`.

Question: q3a's level-boundary clause failed because a refusal can make an omission
illegal but cannot perform the consolidation. Tycho already solves this with a
`LEVEL_COMPLETE` scribe. Ported into the NOOA solver, does the boundary now produce a
**model-written** carry-forward from the winning evidence?

Posture: targeted repair of one failed clause against a recorded incumbent. Not a new
question — q3a answered every other clause and this one alone.

Contrast: q3a, `results/nemo_solver/20260904_011341_ls20_mdfiles_q3a/`.
Held fixed: game `ls20`, variant `mdfiles`, `qwen/qwen3.8-27b`, skill
`interactive-game-solver`, `--visual off`, `--effort-ladder ''`, thinking off via
`ARC_LLM_EXTRA_BODY`, sandbox `inproc --sandbox-cells --sandbox-require
--sandbox-cpu-s 60`, `--max-turns 40`, `--max-env-steps 120`, and every q3a gate
except the one named below.

Changed, two things:
1. **The scribe replaces the z-gate.** q3a's G2 demanded `predicted_path[0]['z']` be
   non-null after a level change. It is removed. In its place the boundary hands over
   evidence and requires a written consolidation.
2. **`--agent-turn-timeout` 300 → 600.** An operational repair explicitly named in
   q3a's permitted next actions; q3a lost 5 of 74 actions to force-advances because
   the actor's own verbosity overran a window sized for q2d.

Change 2 makes the q3a→q3b comparison non-single-factor **for tempo metrics**
(force-advances, wall, tokens per action). It does not touch what P2 measures. Say so
in any claim drawn from this run.

---

## What was ported (frozen)

From `solvers/tycho_nooa/.venv/.../tycho/`:

| Tycho | q3b (NOOA) |
|---|---|
| `agent.py:652` `AgentSpec(id="scribe", trigger=on_events(Event.LEVEL_COMPLETE))` | harness writes `ipc/level_<L>_terminal.json` at the boundary; agent gate holds until consolidated |
| `scribe.user.j2` — compare pre-win frame, winning action, terminal frame | `self.level_terminal(L)` returns exactly those three, plus actions spent |
| "Do not take an action" during consolidation | `submit_actions` returns REJECTED with the scribe instruction |
| always write `notes/level_<L>_insights.md` | `write_notes('level_<L>_insights.md', …)` clears the gate |
| `_write_level_insight_fallback` | `_write_consolidation_fallback` after `SCRIBE_MAX_REFUSALS = 2`; idempotent, never clobbers a real note |
| conditional world-model update | scribe text asks for a helper rewrite *if the evidence changes the model* |

Also added: a `SCRIBE PENDING` line in the per-turn context, so the boundary reads as
a step in the turn rather than as a surprise refusal.

Contestant prior holds: the evidence bundle is frames, an action, and a count. No
object roles, coordinates, goals, or transition rules are named by the host.

## Prediction

| # | metric | q3a | q3b predicted |
|---|---|---:|---|
| P1 | batches >1 whose `predicted_path` scored ≥0.6 | 0.889 | **≥ 0.50** (hold) |
| P2 | completed levels with a **model-written** consolidation | 0.000 | **= 1.00** |
| P3 | re-entry rate over attempted actions | 0.013 | **≤ 0.05** (hold) |
| P4 | waste share of executed env actions | 0.108 | **≤ 0.15** (hold) |

P1, P3 and P4 are hold-the-line clauses: the scribe costs turns, and the failure mode
worth catching is a boundary ritual that displaces play.

**P5 — helper rewritten within 2 agent invocations of a boundary — is RECORDED, NOT
GATED.** q3a gated it at 1.00 and that was a design error: the scribe makes the helper
rewrite *conditional* on the terminal evidence changing the executable model. Gating on
it would demand a rewrite that may not be warranted. It is the more interesting number
and it is not a pass/fail.

Also recorded: fallback invocations (a host-written note is a P2 failure, not a save),
force-advances (q3a: 5), turns consumed by consolidation, wall, tokens per env action.

## Measure

`score_q3b.py`, frozen before launch, in this directory. `score_q3.py` is left exactly
as it scored q3a and is not reused.

- P2: for each completed level L, `team_nemo/shared/knowledge/level_<L>_insights.md`
  exists **and does not contain the literal marker** `Deterministic fallback note:`.
- P5: `write_helper` in an agent invocation within 2 invocations after the boundary step.
- P1/P3/P4 as in `score_q3.py`, including the force-advance-safe frame alignment and
  the attempts-based P3.
- Raw: `ipc/{actions,states,rejections}.jsonl`, `ipc/level_*_terminal.json`,
  `agent_logs/**/events.jsonl`, `team_nemo/shared/{helpers,knowledge}/`, `result.json`.

RHAE is **not** the gate and no score claim may be drawn: at 40 turns and 120 env steps
the reachable game score is capped near 0.10 by our own budget, and the scribe spends
turns on writing.

## Cap

40 agent turns, 120 env steps, `--agent-turn-timeout 600`, 4 h wall, $6 OpenRouter
(q3a spent $2.56). Pre-launch: confirm remaining credit ≥ $6.

## Stop

- Score after ≥15 agent turns.
- Abort on spend ≥ $6, wall ≥ 4 h, `--max-env-steps` reached.
- Refusal deadlock: ≥8 consecutive rejected submits with no committed action
  (`ipc/ABORT`, enforced in code).
- Abort if `reasoning_tokens > 0`.

## Invalid if

- `reasoning_tokens` return on this hatch.
- PDEATHSIG kills the orchestrator (launch via the double-fork).
- Fewer than 15 agent turns — INCONCLUSIVE.
- Game source or id referenced in an emitted cell — VOID.
- **A level completes and no `ipc/level_<L>_terminal.json` is written** — the ported
  mechanism did not fire and P2 measures nothing.
- A gate throws and is silently bypassed (check `launcher.err`).
- No level is completed at all: P2 has no boundary to measure — INCONCLUSIVE on P2,
  while P1/P3/P4 remain interpretable.

## Result map

| result | next action |
|---|---|
| **P2 = 1.00, P1/P3/P4 hold** | The port works and the refusal did not. Retain the scribe; delete the z-gate idea permanently. Next card: depth, at a turn budget where RHAE can move, with its own card. |
| **P2 = 0, note exists only via fallback** | The host consolidated and the model would not. The fallback stays (carry-forward is guaranteed either way), but the boundary is not being *read*. That is the strongest case yet for the thinking-on card, and it is the case q3a could not make. |
| **P2 = 1.00 but P1/P3/P4 regress** | The boundary ritual is displacing play. Keep the scribe, make it cheaper — fold consolidation into the first turn of the new level rather than gating it. |
| **P5 high** | The consolidation is reaching the executable model. Port Tycho's `LEVEL_START` trigger-builder next. |
| **P5 zero with P2 = 1.00** | The model writes prose at the boundary and does not act on it — the same notes-not-machinery failure as q2d, one level up. Do not add a third gate; that finding points at the actor, not the host. |
| **Any Invalid-if** | Do not interpret. New id, new card. Preserve the run. |

## Outputs

`results/nemo_solver/<timestamp>_ls20_mdfiles_q3b/` — `events.jsonl`,
`ipc/{actions,states,rejections}.jsonl`, `ipc/level_*_terminal.json`, `traces/`,
`team_nemo/shared/{helpers,knowledge}/`, `launcher.err`, `result.json`.
Scored by `score_q3b.py`. Failed and VOID runs preserved.

## Standing decisions not reopened

WikiSkill; latent bridge; V16 re-push; thinking-on (its own card, its own hatch, and
this card may strengthen the case for it but does not run it). `score_q2.py` and
`score_q3.py` are left as they ran.
