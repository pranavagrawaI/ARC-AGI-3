# R8 gameplay — official Agent.main, paired arms on public LS20 — locally verified, not frozen, not launched

## What this is

Identity `analysis/actor-critic-construction-kaggle-r8-gameplay-20260915T124458Z/` (card with the
three user decisions filled in). Package `deployment/kaggle/actor-critic-construction-kaggle-r8-gameplay/`
built from this directory by `build.py`. The driver `gameplay_play.py` runs the official
`Agent.main` (pinned ARC-AGI-3-Agents `agents/agent.py`) through the `MyAgent` adapter under OFFLINE
Arcade on the exact `ls20/9607627b` directory, twice: arm A = unmodified v11 worker
(`nooa-join/headless.py`, construction disabled), arm B = the same Actor under the R6/R7 construction
overlay (package `headless.py` with constructor transport). Both arms share one relay, one served
model, one game, one envelope; `ARM-DIFF.json` proves the only differences are the worker command
and per-arm labels/roots, and the driver refuses to play otherwise.

## Envelope (user decision 1; frozen in `gameplay_play.py`, `launch_control.LIMITS`, `MANIFEST.caps`)

600 s model admission and 200 Solver actions per arm; Actor 16384 max tokens; critic 192; HTTP 90 s;
retries 0; reasoning off; arm B constructor 2 requests, 45 s soft / 50 s hard, 30 s per POST;
≤ 259 server chats including 1 warmup (relay caps A 129, B 129, total 258; symmetric by ruling); model readiness 1200 s;
launcher 1800 s; notebook work 3100 s; guardian 3500 + 8 s; reap 30 s; one GPU hour; one push.

## Frozen predecessor files (byte-identical to R7)

`game_preflight.py` (R4), `constructor_contract.py` (R5), `live_agent.py` (R6), `process_guard.py`,
`headless.py`, `constructor.py`, `containment_preflight.py`, `runtime_install.py`; every
`donor_bootstrap.py` function except `main` (probe name, `--launcher-timeout-s 1800`).
Supervisor delta vs R7: the probe-era `min(900, …)` cap removed; claim-ceiling text.
`local_accounting.expected_requests`: whole-run purse (≤ 258 admitted; constructor ≤ 2 per game;
every arm ends by win / step_budget / wall_budget / model_admission_budget / call_cap).

## Verification evidence

- `FINAL-TEST.stdout` / `FINAL-TEST.xml`: 102 passed under mapped-root unshare (R7's 94 → 93 preserved,
  `actor_cap` case became `call_cap`; 9 added). `TEST-INVENTORY.json` lists them.
- `test_runtime.py::test_actual_supervisor_worker_http` (6 cases): the real official loop over both
  arms through the unchanged supervisor branch with a scripted loopback model server; fixture caps
  30 s / 3 actions / 13 calls per arm replace the frozen envelope so both arms finish in seconds.
- `test_gameplay_arms.py`: frozen parameters match the card; arm-equality accepts only the worker
  command and per-arm roots and refuses hidden differences; budget arithmetic; purse enforcement;
  construction activates only in arm B; discriminating control — arm A handed the constructor is
  refused by the relay and the run fails closed.
- `VERIFY.stdout` / `LOCAL-CHECK.json`: 59 assets, 52 compiled sources, 422 protected bindings,
  847 predecessor bindings (R7 appended, never rewritten), no drift.
- `ARM-DIFF.json`: static projection of the frozen bindings (relay port chosen at runtime).
- `LINT.stdout`: ruff clean over this directory, verification-r8 and the changed package files.

```sh
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-r8-gameplay-candidate/build.py
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-r8-gameplay-candidate/tests
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-r8-gameplay-candidate/verify.py
deployment/kaggle/.venv/bin/python -m ruff check --no-cache analysis/actor-critic-construction-kaggle-r8-gameplay-candidate analysis/actor-critic-construction-kaggle-verification-r8
```

## Not done / limits

No freeze, no manifest/authorization, no independent review, no push. The gate
(`verification-r8/launch_control.py check`) HOLDs on the missing run manifest; `make_manifest.py`
refuses on the missing independent package review. Real SGLang, GPU, image, 600 s tempo and the
incumbent critic under live play are unexercised locally (fixture Actor never triggered a critic
review in 3 actions; v10 saw 8 reviews on LS20). Game set is a frozen parameter with `ls20` only;
s5i5/cd82 need their exact public bytes staged for local qualification before the follow-on identity.
