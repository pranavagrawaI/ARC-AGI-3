# Epistemic questioner public candidate

Notebook v11 embeds runtime v8, derived from the operationally qualified v10/v7
public solver. It preserves reasoning-off Qwen3.8-27B-FP8, zero retries, Actor-only
action authority, bounded planning, prediction scoring, and the existing public
game budget machinery.

The critic asks one concise epistemic question at consequential boundaries:
level advance, prediction mismatch, an observed multi-action sequence, three
observed actions, or eight Actor calls since the last valid review. Only
Actor-authored event references are accepted. Empty valid questions consume the
boundary; malformed or unbound output leaves it pending. There is no fixed critic
count budget. Output is capped at 192 tokens.

Qualification commands:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260912-v8/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v11/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v11/verify_repairs.py
```

Qualification is local and uses deterministic or loopback HTTP fixtures. No
Kaggle execution, gameplay benefit, model adoption, or competition score is
claimed. A remote run requires a new immutable experiment card and explicit
authorization.
