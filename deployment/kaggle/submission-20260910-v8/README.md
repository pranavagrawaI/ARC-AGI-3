# Endurance baseline bug fixes — notebook v8

Local repair successor to notebook v7 and runtime v4 from public endurance v2.
This notebook packages `submission-runtime-20260910-v5`. `BUILD.json` binds the
notebook to its runtime contract. The recorded endurance source is unchanged.

From the repository root:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260910-v5/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v8/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v8/verify_repairs.py
```

The verifier uses deterministic fixtures and local HTTP servers. It checks the
actual Agent entrypoints, Python-cell failure, model-request failure, idle
budget stops, process retirement, ordinary actions, contingent continuations,
and the bytes embedded in the notebook. It makes no remote model requests.

The four repaired defects and before/after evidence are recorded in
`../../../analysis/deployment-probes/20260910-endurance-bugfix/REPORT.md`.

Gameplay policy remains the predecessor's: three public games, 600-second model
admission per game, 200 actions, thinking off, zero retries, no summarizer, and
competition reruns refused. The 60-second cell allowance is now monitored by the
separate protocol process. Expiry is an execution failure, not a normal budget
stop. Earlier host deadlines still take precedence.

No upload, new endurance run, or competition submission has been performed.
Local repair checks do not transfer the predecessor's GPU qualification to this
new candidate.
