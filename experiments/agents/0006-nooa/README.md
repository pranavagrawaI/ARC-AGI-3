# 0006-NOOA

`0006-NOOA` is the competence treatment over frozen `0002-NOOA`. Its first
vertical slice makes one action prospectively testable without giving the host
semantic action authority.

The treatment code is intentionally small:

- `competence.py` — archive, neutral evidence operation, claim ledger,
  prediction/reconciliation state machine, and fixed context packet.
- `host/competence_runner.py` — the 0006 host adapter over the frozen 0002
  worker. It carries the commitment through the unchanged final-action channel
  and fails closed before action execution.
- `tests/` — pure protocol and actual offline frozen-worker coverage.

Read [CONTRACT.md](./CONTRACT.md) before adding a mechanism. The candidate and
deletion record for mechanisms held out of 0002 remains in
[NOOA_FEATURES.md](./NOOA_FEATURES.md); executed removal checks are in
[DELETION_LOG.md](./DELETION_LOG.md).

## Reproduce the current slice

Create the frozen worker's ignored local runtime once:

```bash
cd experiments/agents/0002-nooa/nooa-runtime
uv sync --locked --python 3.12
```

Then, from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
uv run --python 3.14 python -m unittest discover \
  -s experiments/agents/0006-nooa/tests -v
```

The end-to-end fixture makes no provider call. It demonstrates protocol and
boundary function only; it is not a score, capability, or treatment-effect
result.
