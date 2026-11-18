# Verdict — 20260904T-nooa-join-public3

Kernel `pranavagrawai/prototype-nooa-join-public-3` version 1. Status ERROR. Elapsed 349.4 s. No retry.

## Classification

**Invalid as gameplay evidence. Valid as a join-packaging failure.**

The official launcher listed `ls20`, `s5i5`, `cd82`, created a scorecard, and loaded all three game classes. SGLang reached readiness. `MyAgent.choose_action` ran. `create_solver` then raised before the NOOA worker process started.

```
KeyError: 'TYCHO_KAGGLE_GAME_UID'
RuntimeError: hardened-host worker identity is missing or invalid
```

`runtime/local_solver.py:_prepare_hardened_session` requires `TYCHO_KAGGLE_GAME_UID` / `GID` when `TYCHO_KAGGLE_HARDENED_HOST=1`. The shared `create_solver` assigns those identities. The NOOA wrapper did not. The R8-L3B outer runtime injects `HARDENED_HOST=1` when it launches the probe, so popping the variable in the notebook cell did not reach `LocalSolver.__init__`.

Zero environment actions. Zero `actions.jsonl`. Three empty-ish session dirs exist only as failed `LocalSolver` construction.

## Card

Success (recording on each game + ≥1 non-RESET action): **not met**.

Result map branch: **negative (load/join) → repair the notebook join; do not open q5.**

## Repair (local, not flown)

`deployment/kaggle/s0-nooa-join-candidate/local_solver.py` now copies the identity assignment from the shared factory. That is a new candidate. It needs a new card and a new push. This run's card forbids retry.

## What this run did prove

- Wheelhouse v11 installs, including `nooa==0.0.1.dev95`.
- Join zip unpacks.
- Official offline list is `ls20,s5i5,cd82`.
- RTX route and SGLang warmup still work.
- Lifecycle containment passed; teardown accepted.

It did not prove CodeAct, `submit_actions`, or any play.
