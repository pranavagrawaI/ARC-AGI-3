# Fixed constructor interface — ready-for-review candidate

Task: **OR-CONSTRUCTOR-SCHEMA-FIX-REPEAT**. Fresh identity `openrouter-constructor-fixed-20260916T214433Z`. No paid calls or credential access during preparation. Prior experiment and all R10 artifacts remain immutable.

## Exact scope

Copied the prior reviewed harness. The only constructor behavior changes are the nested `work_item` schema and targeted validation error text. `fixed-interface.diff` shows this change; `CANDIDATE-DELTA.diff` also shows fresh task identity checks. Provider transport/accounting, reasoning continuation, containment, evaluator, source representation, constructor policies and all limits are unchanged. Both arms receive the same fixed interface. This second descriptive pair jointly changes schema plus feedback; it cannot isolate their individual contribution or be called a replication of the old prompt.

`SCHEMA-TRACE.md` maps every constraint to the inherited validator. No invalid argument is coerced. Schema exposes nine required fields, no extra work_item keys, permitted enums, nonblank strings, nonempty dependencies and references bounded by the actual visible frontier. Existing acceptance of boolean true as step 1 is preserved and explicitly represented; no hidden validator hardening. No source-format repairs or semantic hints were added.

`prefix.jsonl` is unchanged step 1 RIGHT, with the original R10a constructor-visible evidence. Original system/user messages, inspection and sealed ABI replay are unchanged from the previous pair. No historic answer/failure/solution is added to model input. The new feedback is emitted only after that episode actually makes an invalid call.

## Files and immutability

- `EXPERIMENT-CARD.md`, `dispatch.md`: fresh scope, precommitment and authority.
- `INVENTORY.json`: exact review bytes; not an approval.
- `PRIOR-EXPERIMENT-HASHES.json` / `PRESERVATION-CHECK.json`: before/after hashes of prior experiment, plus R10 frozen source checks.
- `SOURCE-BINDINGS.json`, `PREFIX-BINDING.json`, `engine.diff`: inherited R10 source/evidence bindings and transport adaptation.
- `fixed-interface.diff`, `CANDIDATE-DELTA.diff`: change from the prior reviewed harness.
- `test_work_item.py`: actual HTTP/constructor invalid-input, feedback and successful repair-to-sealed-ABI regressions; `test_pair.py`: inherited transport/containment/budget/lifecycle tests.
- `CHECKS.json`, `tests-before.txt`, `tests-after.txt`, `tests-final.txt`: exact commands, hashes, reproduction and qualification.

Copied historical discovery/auth-presence files describe the previous preparation only. They grant no execution authority and no loader was called here. `auth.py` remains byte-identical and inert by default. User permission carries forward for the same sole-key loader only; Hermes owns new release/credential artifacts. No old approval, live directory, consumption or ledger was copied.

## Local checks — no credentials or remote model

From repository root, use a fresh basetemp each time:

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-fixed-20260916T214433Z/test_pair.py analysis/openrouter-constructor-fixed-20260916T214433Z/test_work_item.py --basetemp=analysis/openrouter-constructor-fixed-20260916T214433Z/verification/reviewer
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/openrouter-constructor-fixed-20260916T214433Z/check_candidate.py
```

Fixtures use loopback responses, not a paid model. They execute sealed inspection and initialize/predict/assimilate. Synthetic candidate code belongs only to fixtures, never the live prompt.

## Gates and exact eventual command

Do not execute until PM releases these exact reviewed bytes. Reviewer independently writes decision with task `OR-CONSTRUCTOR-SCHEMA-FIX-REPEAT`, role `independent-reviewer`, decision `PASS`, endpoint `reka/fp8`, inventory_sha256 matching this directory's inventory bytes. Hermes owns release (same task/endpoint/hash; role `Hermes`, decision `RELEASE`) and carried-forward credential gate (same task; role `PM`, decision `APPROVED`, credential_use `sole-key-existing-loader`). No self-written approvals.

After independent PASS, Hermes gates and PM's explicit release, replace the three placeholders with their owners' absolute files and run ONCE:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/.venv/bin/python -B /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/openrouter-constructor-fixed-20260916T214433Z/run_pair.py --execute --review /ABSOLUTE/INDEPENDENT-DECISION.json --release /ABSOLUTE/HERMES-RELEASE.json --credential-approval /ABSOLUTE/CREDENTIAL-APPROVAL.json
```

Parent launches mapped-user workers off then on. Never invoke --worker manually. Any partial `live/` or consumption means HOLD; never reset/retry. Raw wire requests are authoritative for reasoning, not inherited logical engine markers. Each attempt reserves endpoint maximum completion exposure plus conservative input cost. Unknown receipt stops pair; no automatic preflight/retry/fallback. No Kaggle, gameplay, submission or R10b action.
