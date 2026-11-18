# Objection-closure public candidate

Notebook v12 embeds runtime v9. It preserves everything operationally qualified in
v11/v8: reasoning-off Qwen3.8-27B-FP8, zero retries, Actor-only action authority,
event-driven critic triggers, Actor-only source binding, 192-token critic output,
and the public game budget machinery.

Two changes from v11. The critic reads a compact commitment packet (three newest
Actor code items, the last five commitments' rationales as citable claims,
measured transitions as counts and regions, the wall and action budget, and the
objection ledger) instead of the transcript. And one objection is pending at a
time: while it is open the critic is not called and the host restates it locally;
the Actor disposes of it through `submit_actions(closes=...)` with defend,
narrow, withdraw, or probe; three refused commitments expire it.

Qualification commands:

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260912-v9/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v12/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260912-v12/verify_repairs.py
```

Qualification is local and uses deterministic or loopback HTTP fixtures, plus the
72-call saved-trace critic probes recorded under
`analysis/critic-closure-contract-20260912/`. No gameplay benefit, model
adoption, or competition score is claimed. A remote run requires its own
immutable experiment card and explicit authorization.
