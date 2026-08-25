# Matched Actor contract repair v1

This zero-provider checkpoint repairs the five contract seams diagnosed by
`matched-actor-v5-golden-paths`. It preserves the frozen pilot packets and
evidence snapshots while proving that both control surfaces can reach the same
answers with documented operations.

Run from `solvers/tycho_nooa`:

```bash
env -u TMP -u TEMP TYCHO_SANDBOX_RUNTIME=host \
  uv run --frozen python ../../analysis/matched-actor-contract-repair-v1/run.py
```

The run makes no provider calls and no ARC actions.
