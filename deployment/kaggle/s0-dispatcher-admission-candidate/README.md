# Dispatcher package: locally qualified

The delivered notebook, not a loose-source override, passed three actual-worker
process cases: repeated polls and a successor; completion in the receive gap
followed by an identical-grid successor; and two legitimate model calls before
one action submission. The stale state-only notification was skipped after
submission. Explicit-user and mixed-notification controls remain in the earlier
worker qualification.

`qualification.json` records executing source paths, SHA-256 identities and
compiled-code agreement. Only `headless.py` differs inside the embedded ZIP;
the worker, model manifest, dependency, engine-launch and reasoning cells remain
unchanged. `package-manifest.json` binds the delivered notebook and ZIP.

Rebuild/check: `python3 build_package.py --check`.
Replay: `../.venv/bin/python qualify_package.py` (requires the retained local
NOOA/LiteLLM artifacts used by `worker_probe.py`).

Status: **LOCAL_PACKAGE_QUALIFIED**, not Kaggle-integrated qualification.
No GPU speed, scheduler retirement, outer interrupted termination or final
action-boundary supersession claim follows. Do not launch this inherited
long-gameplay notebook as part of the isolated engine comparison.

The tested polling/receive-gap hypothesis is closed for this reachable path.
The reproduced defect is duplicate admission after action submission, not loss
of the original answer. Protocol turn identity, not grid equality, controls the
tested successor behavior.
