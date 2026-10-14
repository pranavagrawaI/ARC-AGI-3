# Planning adoption screen

One private Kaggle session tests control, bounded planning, prospective probing,
and both helpers at two restored s5i5 checkpoints and two sampling seeds.
The [experiment card](EXPERIMENT-CARD.md) fixes the contrast and limits.

Accepted identity: kernel **138688264**, version **1**,
[planning-adoption-20261014](https://www.kaggle.com/code/pranavagrawai/planning-adoption-20261014).
The notebook, package and card are frozen. `launch.py` and `build.py` refuse a
second attempt. There is no competition submission or promotion.

## Evidence

`identity.json` records all 96 packaged file hashes and the notebook hash.
`evidence/qualification.json` records actual packaged native fixtures, a strict
HTTP constructor/tool-loop fixture, one-call error handling, and trickling-body
deadline tests. These fixtures prove local execution and transport behavior,
not live efficacy. `evidence/launch-response.json` records acceptance.

`status.py` binds the remote kernel, version and notebook cells before polling.
After termination, `collect.py` binds the same identity before and after downloading
current-session receipts, then records file sizes and SHA256 hashes.
`analyze.py` verifies those bytes, the complete matrix, native action replay,
request/response coverage and budgets, the declared first request, effective
serving settings, engine identity, and protected source identities.

The raw output directory is ignored by Git. `collection.json` records its paths
and hashes. [RESULT.json](RESULT.json) records all sixteen valid episodes and the
failed treatment gates. [FINDINGS.md](FINDINGS.md) explains the absence of helper
adoption and the successful control trace. [MECHANISM-EVIDENCE.json](MECHANISM-EVIDENCE.json)
binds the selected excerpts to raw file hashes and call numbers. The root
`matrix.json` is a byte-identical copy bound to the frozen package manifest.

## Read-only reconciliation

From the repository root:

```bash
/home/pranav/.local/share/uv/tools/kaggle/bin/python -B analysis/planning-adoption-20261014/status.py --stream
```

Only after the session is terminal:

```bash
/home/pranav/.local/share/uv/tools/kaggle/bin/python -B analysis/planning-adoption-20261014/collect.py
solvers/tycho_nooa/.venv/bin/python -B analysis/planning-adoption-20261014/analyze.py
```

`collect.py` may retry read-only downloads. It cannot relaunch the kernel.
Use the Python 3.12 solver environment for native replay; the separate Python
3.13 environment lacks `imageio` for this restored actor.

## Scope

This is a selected-checkpoint adoption/completion screen in one game. At least
two paired gains with no losses and all four valid pairs are needed to pass a
treatment. Even a pass would require a different-game/full-game comparison
before a score claim. Helper availability, model-valid plans and isolated
prediction matches do not establish gameplay efficacy by themselves.
