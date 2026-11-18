# LS20 sparse-reasoning public candidate v1

Notebook v15 embeds runtime v11 and runs only LS20. Routine Actor calls have Qwen
thinking disabled; the existing `deep_investigate` activation may enable one
2048-token reasoning call for a hard abstraction gap.

Qualification commands:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260912-v11/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v15/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v15/verify_repairs.py
```

Local results are recorded under
`analysis/deployment-probes/20260912-ls20-reasoning-local-v1/`.
No Kaggle execution under this package yet. A remote run requires its own
immutable experiment card and explicit authorization.
