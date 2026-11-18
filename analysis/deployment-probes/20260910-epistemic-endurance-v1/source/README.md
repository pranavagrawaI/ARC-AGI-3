# Epistemic contract endurance candidate

Notebook v9 embeds runtime v6, derived from the locally repaired v8/v5 baseline.
Ordinary exploration does not require a goal, helper, or hypothesis pair. The
Actor guidance preserves evidence scope and unresolved conditions; typed
experiment results explicitly certify measured predictions only. Behavioral
adoption must be assessed from the live trace, not inferred from prompt text.

`plan_actions(target, actions=None, limits=None)` searches the Actor's own
executable model in a disposable POSIX main-thread fork. Loading, encoding,
prediction, target evaluation, and fresh replay are contained by the parent
wall deadline. Missing containment fails closed. Search results distinguish
complete model exhaustion from cutoff/error. Linked submissions reject stale
state/helper identities and still require all normal batch gates.

Qualification (local fixtures and loopback HTTP only):

```bash
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-runtime-20260910-v6/freeze_candidate.py --check
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v9/build.py
deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v9/verify_repairs.py
```

Run evidence and the immutable launch card:
`analysis/deployment-probes/20260910-epistemic-endurance-v1/`.
One public-only Kaggle execution: LS20/S5I5/CD82, 600s admission/game,
200 actions/game, thinking off, zero retries, no summarizer or competition
submission. No controlled causal contrast is claimed with the historical run.

The planner is a computation-containment tool, not proof of good epistemics.
Source mutation, timeout, and result provenance are structurally checked;
semantic correctness and appropriate uncertainty still require trace review.
