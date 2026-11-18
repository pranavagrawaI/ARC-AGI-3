# q3a-host-control-loop

Written 2026-09-04, before any implementation and before the run. Frozen at launch.
Companions: `../20260903T-skill-following-q2/{perfect-approach.md,flaws.md,PRECOMMIT.md}`
and the q2d run dir named under Contrast.

Question: With thinking still off, does moving the control loop out of the actor —
host refusals, host bookkeeping, and a chassis that stops affording notes-then-spray —
produce **compilation** (a searched path that constrains the batch; a helper that
refreshes at a level boundary) and **recovery** (a recorded death that is not re-entered)
on `qwen/qwen3.8-27b`?

Posture: bundled treatment against a recorded incumbent. Diagnosis of q2d is closed
(`flaws.md`). This run tests whether the diagnosed fixes change the actor's control,
not which fix did it.

Contrast: `../20260903T-skill-following-q2/results/nemo_solver/20260903_160733_ls20_mdfiles_q2d/`
Held fixed: game `ls20`, variant `mdfiles`, model `qwen/qwen3.8-27b`, skill
`interactive-game-solver`, `--visual off`, `--effort-ladder ''`,
`ARC_LLM_EXTRA_BODY={"reasoning": {"enabled": false}}`, sandbox `inproc --sandbox-cells
--sandbox-require --sandbox-cpu-s 60`, `--agent-turn-timeout 300`, `--max-turns 40`,
`--max-env-steps 120`.
Changed: the whole change set below, at once.

**This is a bundle, not a single factor.** A positive result licenses "this bundle
changes control on this executor" and nothing narrower. Component attribution requires
ablations that are not authorised here and are only worth running if the bundle passes.

---

## Change set (frozen at launch; new candidate id, not inside V15 bytes)

Implementation site is one batch loop plus the strings around it.

### Host — `harness.py`

| # | change | site |
|---|---|---|
| H1 | `diff_summary` scoped to the **turn**: capture `turn_start_grid` before the batch loop; keep a per-action list alongside it. Today `prev_grid` is reassigned inside the loop, so a 10-action turn reports action 10 and the doc claims the turn. | `:363`, `:539` |
| H2 | No-impact truncation: after a step, if its changed cells are empty or lie wholly inside the discovered HUD band, execute no further actions this batch and say so in the note. | after the existing checks at `:616-630` |
| H3 | HUD band discovered online: rows changing on ≥0.8 of observed pairs, warm-up 5 eligible frames. Before warm-up only a zero-change step truncates. No colour or row index is named. | new helper |
| H4 | Observation record: append-only, agent-readable. Blocked `(cell, action)` pairs, terminal states with the action that reached them, the HUD band, and the spawn observed at each level. Handed to the agent each turn as structured data. | new |
| H5 | Re-encode gate: on `levels_completed` change, reject submits until an `encode` has run against the new grid. | `:624` + agent side |

**Ordering constraint, not negotiable.** H2 runs strictly after the existing
`WIN` → `GAME_OVER` → `level_completed` breaks. Step 103 of q2d was HUD-only *and*
GAME_OVER; a predicate placed ahead of the terminal checks swallows a death as a no-op.

### Agent — `solver_agent.py`

| # | change | site |
|---|---|---|
| A1 | `submit_actions(actions, rationale, predicted_path=None)`. A batch longer than 1 without a `predicted_path` of matching length is REJECTED. The path is recorded and compared, per action, against the next state. | `:896` |
| A2 | A batch longer than 1 is REJECTED while the last retrodiction failed. Batch length is earned by search, not by confidence. | `:896` |
| A3 | Status line gains `helpers: world_model.encode,predict` beside the existing `notes:` line. Today the world model is invisible and markdown is on the dashboard. | `:840-857` |

### Signifiers — text only, no behaviour

| # | change | site |
|---|---|---|
| S1 | System prompt leads with "the plan is `self.h.*.predict`; notes are the lab book", not "your accumulated knowledge lives in markdown files". | `:1119` |
| S2 | Per-turn workflow: grid and helpers **before** the knowledge store, and `rationale="predict: ..."` to match the skill. The current `"prediction..."` is why retrodiction scored 0/24 against the skill's string. | `:1044-1050` |
| S3 | `submit_actions` docstring example becomes a single action; states that batch length is earned. The current example is a four-action spray under a cap of 20. | `:897-912` |
| S4 | Best effort: raise `<arc_skill>` above the CodeAct `strategy_prompt`, or, if context ordering is fixed by the framework, insert a short ARC override immediately after it. The skill currently opens at 85% depth of a 28,829-byte round-0 prompt, after three pointers telling the actor to follow a block it has not yet read. | context assembly |

S4 is best-effort and **not gated on**: if the framework's ordering cannot be changed
without patching third-party code, the override block ships instead and the card is
still valid. Record which of the two shipped.

Contestant prior (`CONTEXT.md`) holds throughout: nothing here encodes game-specific
object roles, coordinates, visual conventions, goals, resources, or transition rules.
H3 and H4 are observational — "rows that change on almost every step", "cells that
absorbed a move" — in the same category as the HUD discovery already accepted offline.

---

## Prediction

Directional, against q2d's recorded values.

| # | metric | q2d | q3a predicted |
|---|---|---:|---|
| P1 | batches of length >1 carrying a `predicted_path` whose per-action prediction matched the observed step on ≥0.6 of its entries | 0 of 18 | **≥ 0.50 of such batches** |
| P2 | helper rewritten within 2 agent turns of a `levels_completed` change | 0 of 1 | **1.0 of level-ups** |
| P3 | submitted actions targeting a `(cell, action)` already in the host observation record as blocked or terminal | present (turn 21→22) | **≤ 0.05 of submitted actions** |
| P4 | waste share of executed env actions (HUD-only ∪ blocked) | 0.28 | **≤ 0.15** |

P4 is the weakest of the four. Truncation removes tails mechanically — the offline
counterfactual on q2d's own trajectory removed 29 of 123 actions — but A2 pushes early
play to batch size 1, where a no-impact step is always first in its batch and cannot be
truncated. A high P4 with a healthy P1 is not a failure of the bundle.

**Named counter-outcome — compliance without competence.** The refusals bind, median
batch length stays 1 for the whole run, and `levels_completed ≤ 1`. This is the outcome
where we taught the actor to satisfy the host and nothing else. It is a distinct result
with its own branch below, not a variant of failure.

## Measure

Scorer `score_q3.py`, written **and frozen before launch**, in this directory. It reads
only run artifacts:

- `agent_logs/**/events.jsonl` — `agent_turn`, `env_step` (grids for P3/P4 labels)
- `ipc/actions.jsonl` — batches, `rationale`, `predicted_path`
- `ipc/states.jsonl` — `action_results`, per-action `diff_pixels`, `levels_completed`, notes
- `<workspace>/helpers/world_model.py` mtimes and `write_helper` events — P2
- the H4 observation record — P3
- `result.json` — turns, `game_overs`, `levels_completed`, RHAE, termination
- `traces/` — `reasoning_tokens`, spend, latency

The unit for P1 and P2 is the **agent turn**, as in the Q2 precommit. Do not re-denominate
to env steps; that made the Q2 gate unpassable for a following actor and the same trap
applies here.

Also reported, not gates: output tokens per committed env action (q2d 787; Son Pham's
death line 644), wall, spend, cache-read share, per-level RHAE.

**RHAE is not the gate and cannot be.** `rhae.py` gives `game = Σ(i·levelᵢ)/Σi`; ls20 has
7 levels, Σi = 28, so the first level is worth 1/28 and the score is governed by how deep
a prefix clears. Level 1's human baseline is 22 actions; seven levels is plausibly 150–250.
At `--max-env-steps 120` the reachable game score is capped near 0.10 by our own cost
control. Comparability with q2d is worth more here than depth, so the cap stays at 120 and
**no claim about game score may be drawn from this run.** A run that asks whether any of
this helps the score needs a larger step budget and its own card.

## Cap

- 40 agent turns, 120 env steps, `--agent-turn-timeout 300`
- 4 h wall
- **$6 OpenRouter.** q2d spent $0.86; refusals add rejected round-trips, so budget 3–5×.
- Pre-launch gate: confirm remaining credit ≥ $6 on the key (handoff estimate ≈ $8.13 of a
  $10 limit). Do not log the key.

## Stop

- Score after ≥15 agent turns; scorer PASS/FAIL ends interpretation, not the run.
- Abort on spend ≥ $6, wall ≥ 4 h, or `--max-env-steps` reached.
- **Abort on refusal deadlock:** ≥8 consecutive REJECTED submits with no committed env
  action. The refusals are new machinery and can starve the run; this is their kill switch.
- Abort if `reasoning_tokens > 0` on any call.

## Invalid if

- `reasoning_tokens` return on this hatch (thinking-off is the actor we are measuring).
- `run_solver.py`'s `PR_SET_PDEATHSIG` kills the orchestrator — launch via
  `daemonize_q2d.py`'s double-fork, not `setsid nohup`. A run that dies this way is VOID.
- Fewer than 15 agent turns — INCONCLUSIVE, do not interpret.
- Game source or game id referenced in an emitted cell — VOID.
- A host refusal throws and is silently bypassed (check `launcher.err` and the note stream);
  the run then measures a different, unspecified treatment.
- H1 or H2 is not actually live in the run — verify before interpreting: `diff_summary` on
  a multi-action turn must match the whole-turn cell count, and at least one truncation note
  must appear if any no-impact step occurs mid-batch.

## Result map

| result | next action |
|---|---|
| **P1 ∧ P2 met** (compilation) | The control loop belongs in the host. Retain. Next card: the same bundle at a step budget where depth is reachable, scored on RHAE. Component ablations become worth their cost. |
| **P1 ∧ P2 met, P3 ∧ P4 missed** | Compilation without recovery. The observation record is being read and not obeyed. Next is the goal predicate — the frame one action before a level completion — not more refusals. |
| **P3 met, P1 missed** | The host is doing the work and the actor is not compiling. Retain the host bookkeeping, drop the `predicted_path` gate as ceremony, and open the thinking-on card: search is the step we cannot get from a thinking-off 27B. |
| **Counter-outcome: median batch 1, `levels_completed ≤ 1`** | The bundle bought compliance, not competence. Do not add refusals. Reopen posture: this is the case for reasoning-on, on its own card and its own hatch. |
| **Any Invalid-if fires** | Do not interpret. New run id, new card. Preserve the dead run. |
| **All four met and `levels_completed ≥ 2`** | Strongest branch available under this cap. Retain, and the WikiSkill/B2 card — writable since q2d, still not next — moves ahead of the ablations. |

## Outputs

`analysis/local-probes/20260904T-host-control-loop-q3/results/nemo_solver/<timestamp>_ls20_mdfiles_q3a/`
— `events.jsonl`, `ipc/{actions,states}.jsonl`, `traces/`, `team_nemo/shared/{helpers,knowledge}/`,
the H4 observation record, `launcher.err`, `result.json`, `gameplay.json`.
Scored by `score_q3.py` in this directory. Failed and VOID runs are preserved, not deleted.

---

## Standing decisions this card does not reopen

WikiSkill / evolved `SKILL.md` — writable since q2d, still not next; the failures are
control, not skill prose. Latent bridge — unreachable on one Qwen-3.8-27B by 30 Sep.
V16 vision re-push — answered locally. Thinking-on — a live hypothesis for compilation,
its own card, its own hatch; it must not enter this bundle as a silent `extra_body` change.
`score_q2.py` — left as it ran; not retuned to any result.

---

## Pre-launch amendments (2026-09-04, before any execution)

Recorded because implementation forced choices the card had left implicit. Made
before the run started and before any q3a data existed. Nothing here was chosen
after seeing a result.

1. **P3 is counted over attempts, not executions.** The host now refuses a
   re-entry at submit time (gate G1), so scoring only executed actions would make
   P3 vacuously 0 and measure nothing. Refused submits never reach
   `ipc/actions.jsonl`, so `submit_actions` logs every rejection to
   `ipc/rejections.jsonl` (turn, step, reason, actions) and
   `P3 = (executed re-entries + refused re-entry attempts) / (all attempted actions)`.

2. **HUD discovery threshold is 0.9, not the offline probe's 0.8.** On the q2d
   frames rows 61–62 change on 0.984 of transitions and the next row on 0.228, so
   every threshold in 0.7–0.95 returns the same band. 0.9 is free on the only real
   data we have and narrows the chance of absorbing a frequently-moving playfield
   row on an unseen game. Host and scorer use the same value; the scorer
   rediscovers the band from the frames rather than trusting the host's record, so
   a bug in the host machinery cannot score itself as a success.

3. **The refusal-deadlock abort is code, not prose.** `submit_actions` counts
   consecutive rejections since the last committed batch and writes `ipc/ABORT` at
   8; the harness checks that sentinel at the top of its loop and terminates with
   `termination = "aborted"`. Without this the Stop condition would have been a
   sentence nobody enforces.

4. **S4 shipped as a system-prompt override, not context reordering.** The
   framework assembles `<arc_skill>` after the CodeAct `strategy_prompt` and the
   ordering is not ours to change without patching third-party code. The system
   prompt renders first, so the override lives there and names each conflicting
   CodeAct instruction explicitly (`return_result`, "use LLM reasoning", "keep
   cells short", the second sense of "helpers", `await`). This is the fallback the
   card permitted; record it as the one that shipped.

5. **Gate G1 does not fire when every available action is already known bad** from
   the current state, or the run would starve rather than explore.

## Incumbent, measured with the frozen scorer

`score_q3.py` run against the q2d trajectory, saved as `incumbent-q2d-score.json`.
All four metrics fail on the incumbent, which is the point — they discriminate.

| | q2d | q3a threshold |
|---|---:|---|
| P1 search-backed batches | 0.000 (of 16 batches longer than 1) | ≥ 0.50 |
| P2 helper refresh on level-up | 0.000 | ≥ 1.00 |
| P3 re-entry rate | 0.220 (27 of 123) | ≤ 0.05 |
| P4 waste share | 0.285 | ≤ 0.15 |
| median batch | 5.0 | — |
| discovered HUD band | rows 61–62 | — |

One observation from that scoring run, recorded before q3a exists: **q2d step 103
— the GAME_OVER — is itself a re-entry.** The actor submitted `LEFT` from a
playfield state in which `LEFT` had already been observed to change nothing. Gate
G1 refuses exactly that submit. This is a property of the recorded incumbent, not
a prediction about q3a: after the first refusal the trajectory diverges and no
counterfactual survives it.
