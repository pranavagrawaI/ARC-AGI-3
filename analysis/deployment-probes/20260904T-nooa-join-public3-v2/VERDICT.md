# Verdict — 20260904T-nooa-join-public3-v2

Kernel `pranavagrawai/prototype-nooa-join-public-3` version 2. Status ERROR. Elapsed 4278 s. `launcher_returncode` 0. No retry.

## One-factor (v1 crash)

**Repaired.** `result.json` error does not contain `TYCHO_KAGGLE_GAME_UID`. All three games constructed a worker with `chassis=q3c-mdfiles-after` and `command=.../headless.py`.

## Play

The official loop finished 16 frames on each of `ls20`, `s5i5`, `cd82`. Levels completed: 0.

NOOA `ipc/actions.jsonl` (host-side submit_actions):

| game | batches | submitted env actions | first batch |
|---|---|---|---|
| ls20 | 7 | 20 | `LEFT` |
| s5i5 | 16 | 16 | `RESET` (model-authored probe) |
| cd82 | 15 | 17 | `UP` |

Helpers were written (`ls20` `model.py`, `cd82` `world_model.py`, `s5i5` `worldmodel.py`). Rationales are `predict: ...` with a predicted_path.

ls20 official recording grids change on all 16 frames, so the environment actually stepped. The probe then scored `total_solver_actions=0` because every recording `action_input.id` is `0`. That field is not the action that moved the grid. rc=73 is the probe, not the actor.

## Card

Stop condition: GAME_UID absent **and** ≥1 non-RESET action in the set.

- GAME_UID absent: yes.
- Non-RESET action: yes (`LEFT`, `UP`, `CLICK 39 21`, `DOWN`/`SPACEBAR` in ipc; ls20 grids change every frame).

Result map: **positive** — freeze this join as the Sep 10 article base. Next is packaging/scorecard, not q5.

Kernel ERROR does not overturn that. It is a bad measure in `nooa_public_probe.summarize_recordings`, which treated `action_input.id==0` as "no play." Do not burn another GPU hour to re-score the same bytes.

## Claim ceiling

This establishes that the q3c NOOA chassis, on the R8-L3B Kaggle route, can boot, call the local Qwen server, submit actions, and move public grids. It does not establish RHAE, a hidden-half score, or that 16 actions is a useful budget.
