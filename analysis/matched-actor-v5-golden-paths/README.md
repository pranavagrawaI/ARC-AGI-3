# Matched Actor V5 golden paths

This package executes answer-key-independent, zero-provider-call golden paths
through the exact V5 Native and CodeAct controllers for the five incomplete paid
packets. Planning reads only each task and its frozen evidence; the held-out
answer key is used afterward to score the returned value.

The proof records runtime-minimum provider operations, whether each path fits the
six-call envelope, and whether the successful path can be derived from the
model-visible contract rather than undocumented runtime knowledge.

Run from `solvers/tycho_nooa`:

```bash
env -u TMP -u TEMP TYCHO_SANDBOX_RUNTIME=host \
  uv run --frozen python ../../analysis/matched-actor-v5-golden-paths/run.py
```
