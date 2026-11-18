# LS20 reasoning-on public candidate v1

Notebook v14 embeds runtime v11 and runs only LS20 with Qwen thinking enabled.
The purpose is to measure terminal-class predictions and false host-event claims
against exact preserved provider-bound prompts, not to claim a score improvement.

Qualification commands:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260912-v11/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v14/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v14/verify_repairs.py
```

Local results are recorded under
`analysis/deployment-probes/20260912-ls20-reasoning-local-v1/`.
No Kaggle execution under this package yet. A remote run requires its own
immutable experiment card and explicit authorization.
