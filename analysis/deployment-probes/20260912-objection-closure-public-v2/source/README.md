# Objection-closure public candidate v2

Notebook v13 embeds runtime v10. See the runtime README for the two removals
made after the v9 public run: the Actor cannot end a game, and the critic packet
carries no projected action count.

Qualification commands:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260912-v10/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v13/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v13/verify_repairs.py
```

Passed: freeze-check, 119 unit, 35 entrypoint, 4 worker
(`analysis/deployment-probes/20260912-objection-closure-local-v2/VERIFIED.json`).
No Kaggle execution under this package yet. A remote run requires its own
immutable experiment card and explicit authorization.
