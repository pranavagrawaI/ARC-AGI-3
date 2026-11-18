# q4 result — 20260904_165006_ls20_mdfiles_q4

Card: `../../../EXPERIMENT-CARD.md`. Scorer: `../../../score_q4.py`.
Contrast: q3c, `../../../../20260904T-host-control-loop-q3/results/nemo_solver/20260904_140403_ls20_mdfiles_q3c/`.

## Validity

Not invalid. `reasoning_tokens = 0`. 59 recorded agent turns (floor 15). Orchestrator
survived to the turn cap (ppid 1167 `systemd --user` throughout). No `ipc/ABORT`.
No game id in an emitted cell checked at launch (alias `game-e6d785`). Terminated
on `max_turns` (60 / 60; 197 of 200 env steps).

The treatment never fired: `ipc/goal_candidate.json` was not written, because
no level completed. That is the card's INCONCLUSIVE branch, not an Invalid-if.

## Gate

| clause | q3c | **q4** | threshold | |
|---|---:|---:|---:|---|
| P1 search-backed batches | 0.944 | **0.949** | ≥0.50 | pass |
| P2 model-written consolidation | 1.000 | **INCONCLUSIVE** | =1.00 | no level |
| P3 re-entry rate | 0.020 | **0.033** | ≤0.05 | pass |
| P4 waste share | 0.064 | **0.036** | ≤0.15 | pass |
| P6 discriminator before deadline | 0.000 | **INCONCLUSIVE** | =1.00 | no transfer |
| P7 revision after miss/expiry | 0.000 | **N/A** | =1.00 | no miss |

**VERDICT: INCONCLUSIVE.**

P5 (recorded): no level-up, unmeasured. P8: unmeasured. `<truncated>` cells: 0.
Force-advances: 0.

## Observation — the port had nothing to score

q3c cleared level 0 in 32 actions and that is when the candidate record would
have opened. This run spent **197 actions and 60 turns on level 0** and never
crossed. Spend **$3.56** of the $6 cap (start remaining $6.32 → $2.75). Wall
5600 s.

P1 held (0.949 of 39 long batches). The actor was still submitting
search-backed routes, not spraying. One batch in 59 carried a `level_up`
expect — turn 2, **8** actions, "UP x7 to reach goal 9-spiral" — longer than
the discriminator cap of 5, so even if a candidate had been open it would
not have bound. Every later batch was `move`/`no_impact`. Last rationales
were still "Navigate to entity zone (30,19)".

The control loop q3c earned did not collapse. The frontier-4 mechanism did
not get a trial.

## Economy

| | q3c | q4 |
|---|---:|---:|
| agent turns | 38 | 59 |
| env actions | 202 | 197 |
| levels completed | 1 | **0** |
| median batch | 2.0 | 5.0 |
| P1 | 0.944 | 0.949 |
| force-advances | 1 | **0** |
| spend | $3.68 | **$3.56** |
| wall | — | 93 min |
| reasoning_tokens | 0 | 0 |

RHAE 0.0. No score claim.

## Permitted next action

Card result map, **P6/P7 INCONCLUSIVE (no level)**:

> Stop running this experiment. q3a reached a boundary, q3c reached a
> boundary, a third miss means the question is again "why can this actor
> not find a goal", which q3b already answered and q3c showed is not a
> dynamics problem. Different card, not a larger budget.

Do not run a q4b that only raises `--max-turns` or `--max-env-steps`.
The goal-candidate record is untested, not falsified. Retain the q3c
scribe and control loop. The next card, if any, is not this line.
