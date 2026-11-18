# q5-seed-live-ls20

Written 2026-09-04, before launch. Offline probes cannot locate actor
failures. This run is that contact.

Question: With `symmetry.py` seeded and discoverable (C1–C4), does this
actor **execute** a reach (`self.h.symmetry.*` or `symmetry.*` in a cell
that ran)? If it does, does any such turn also file a ≤5-action
`level_up`/`terminal` path?

Posture: diagnosis. One factor: the seed is present. Not a claim that
ls20's win is congruence.

Contrast: q4, `../20260904T-goal-candidate-q4/results/nemo_solver/20260904_165006_ls20_mdfiles_q4/`
(same chassis, no seed, S1 = 0 by construction). Held: ls20, mdfiles,
`qwen/qwen3.8-27b`, thinking off, sandbox, timeout 600. Changed: seed +
status-line `provided:`.

Prediction: S1 ≥ 1 is the live uncertainty (Tycho uptake called it at
turn 3; this chassis has never had the module). S2 ≥ 1 is unlikely in
25 turns — q4 never left L0 in 60 — and is recorded, not a pass/fail
for the seed.

Measure:
- S1 = executed reaches from `reach_scorer.py --module symmetry` ≥ 1
- S2 = a turn with an executed reach **and** a submitted batch of length
  1–5 whose `predicted_path` has `level_up` or `terminal` (recorded)
- `ipc/seeded_helpers.json` digest `7e0c1f1c…`
- no `seeded_helper_overwrite` in rejections
- P1–P4 recorded, not gated (short cap)

Cap: **25 agent turns**, 80 env steps, 2 h wall, **$2.50** OpenRouter
(remaining $2.75). `--agent-turn-timeout 600`.

Stop: score after ≥5 agent turns. Abort on spend ≥ $2.50, wall, env-step
cap, `ABORT`, `reasoning_tokens > 0`.

Invalid if: digest mismatch or no `seeded_helpers.json`; seed path not
`symmetry.py`; `reasoning_tokens > 0`; <5 turns; PDEATHSIG; game id in a
cell.

Result map:
| result | next |
|---|---|
| **S1 = 0** | Discoverability failed. Do not rerun ls20. Next is a stronger pointer or `read_helper` in the workflow — own card. |
| **S1 ≥ 1, S2 = 0** | They compared. They did not bet it was the goal (or never saw a pair). Inspect which figures. Next is q6 (goal-form hint) or ar25 when the env is local — not more ls20 at this cap. |
| **S2 ≥ 1** | Retain the seed. Next is a sham module, own card. |
| **Invalid** | Do not interpret. New id. |

Outputs: `results/nemo_solver/<timestamp>_ls20_mdfiles_q5/`
