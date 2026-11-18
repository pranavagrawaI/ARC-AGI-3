# Qualification result — 2026-09-05

The repair satisfies the local host, detector, resolution, scoring and packaging contract. No Kaggle kernel, paid inference, commit or push was performed. The acceptance evidence is CPU qualification with synthetic observations and fake model replies, including real NOOA Actor/CodeAct and real adapter-to-worker subprocess paths.

Candidate contract SHA-256: `1d65e3669ff7ace45eb271aaf943dff35fc279a6ebeba95912e89884d3a05459`.

## Checks actually run

From the repository root:

```bash
deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/tests/test_q6_host_contract.py deployment/kaggle/tests/test_q6_candidate.py analysis/local-probes/20260905T-stall-authority-q6/test_q6_contract.py analysis/local-probes/20260905T-stall-authority-q6/test_integrity_guard.py -q
```

Result: **27 passed**, two existing `set_dynamic` deprecation warnings, 15.04 seconds. Covers native observation/scoring parity, boundary-tail identity, no-impact stops, prediction evidence, game-over/action-budget lifecycle, killed-worker cleanup, same-invocation rejection repair, treatment forwarding, detector parity, resolution validation, terminal censoring, tamper rejection, scribe evidence, episode aggregation, independent identities, integrity and complete deterministic notebook payloads. The test interpreter is the Kaggle deployment's Python 3.12 environment.

From `deployment/kaggle`:

```bash
.venv/bin/python -m pytest tests/test_my_agent.py tests/test_nooa_join.py tests/test_local_solver.py tests/test_join_qualification.py -q
```

Result: **45 passed, 1 skipped**, 36.67 seconds. These existing checks protect adapter/runtime behavior and the archived s0 join. They do not qualify q6 by themselves.

From the repository root:

```bash
python3 analysis/discrimination-replay/sweep.py
python3 deployment/kaggle/q6-nooa-candidate/freeze_candidate.py --check
python3 deployment/kaggle/q6-nooa-candidate/build_notebook.py --output deployment/kaggle/q6-nooa-candidate/control.ipynb
python3 deployment/kaggle/q6-nooa-candidate/build_notebook.py --treatment --output deployment/kaggle/q6-nooa-candidate/treatment.ipynb
```

Replay: correct separation direction at **27/27 settings**, 4.89x at the selected setting, range 2.37x–17.38x. This is historical association, not evidence that treatment works. Both notebooks built and validated; manifest verification passed. Ruff checks passed on new shared modules, scorer, qualification driver, candidate Python files and q6 tests. Canonical solver and native harness passed `py_compile`.

```bash
deployment/kaggle/.venv/bin/python analysis/local-probes/20260905T-stall-authority-q6/qualify_q6.py --output analysis/local-probes/20260905T-stall-authority-q6/qualification/20260905-final-contract
python3 analysis/local-probes/20260905T-stall-authority-q6/score_q6.py analysis/local-probes/20260905T-stall-authority-q6/qualification/20260905-final-contract/control analysis/local-probes/20260905T-stall-authority-q6/qualification/20260905-final-contract/treatment --output /tmp/q6-final-rescore.json
```

Both scripted sessions scored from raw files: 51 actions, one detected stall each. Treatment emitted detection, review, host consequence and authored resolution; its triggered falsifier led to a changed basis. Both P9 values are 1. The arm is **INCONCLUSIVE: scripted qualification is not a research arm**. Missing completion evidence also leaves P2 unqualified. Saved fixtures retain raw states, transitions, submissions and authority records plus candidate provenance. A fresh CLI rescore matched the saved report.

## Problems found during implementation

- Native history and level-boundary semantics were absent in the Kaggle join. Extracting the native evidence owner and testing actual action identities removed the stale queue tail and restored gate inputs.
- A real CodeAct readback reproduced slash mutation caused by root-path redaction. The root prefix was removed and the request-boundary guard was tested against modified feedback. Checking assistant source alone would have missed it.
- Publication order allowed the host to act before the authority record existed. The record now precedes the action notification; observations likewise precede state notifications.
- Initial notebook packaging omitted `stub_agent.py` although startup verifies it. Payload tests now check every manifest file, not a hand-maintained subset.
- Replay initially treated an automatic reset as an ordinary native observation. It now follows the native reset convention.
- Initial test invocations exposed mixed pytest import roots and a CodeAct test waiting correctly for the next host state. Fixtures now establish the import path and cancel the waiting actor after the asserted action; these were test-harness issues, not gameplay fixes.

Independent spec and standards reviews informed the repairs. The final narrow provenance review found no blocker and reproduced rejection of copied run/session identities.

## Limits and next action

The next trial is the bounded Kaggle preflight described by `../EXPERIMENT-CARD.md`: verify actual FP8 model behavior, zero reasoning tokens, complete artifacts, runtime/quota feasibility and sufficient evaluable stall incidence. Those facts cannot be established by these fake-model checks. Do not interpret the scripted result as launch qualification or a treatment effect.

Use the new candidate, not s0. Recheck its manifest before running; any source change requires regeneration and renewed qualification. Keep Kaggle and native results separate. Integrity coverage is explicitly limited to complete note-API printed readbacks; unobserved coverage is reported. GPU execution, performance, actual RHAE/spend availability and research power remain untested. No broad repository-wide test result is claimed.

Resume from `../CHANGES.md`, `../EXPERIMENT-CARD.md`, `../score_q6.py`, the saved `../qualification/20260905-final-contract/SCORE.json`, and the candidate's `CHASSIS.json` plus control/treatment notebooks.
