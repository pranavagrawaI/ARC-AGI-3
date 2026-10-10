# Fixed-capture Marlin GPU diagnostic

One authorized diagnostic replay of the nine saved v3 inputs. The original v3 numerical test is retained as a recorded result; finite rejections no longer suppress timing. This run grants no serving advancement.

- Eight repetitions of control, treatment and an independently allocated second control graph, on original/changed/restored inputs.
- FP32 and separate BF16-boundary diagnostic references, with all raw outputs retained.
- Thirty AB/BA pairs each for graph and eager control-treatment timing, plus an independent-control graph timing null.
- Same hash-qualified wheel, same reduction patch and same captured tensors. No server load, recapture, ARC gameplay, dataset upload or scoring.
- One RTX PRO 6000, one push, 3600-second hard cap. No automatic retry.

The prepared v4 in another directory remains unchanged; it is a different experiment with relaxed gates and new captures. This diagnostic uses a fresh identity.

Run local checks:

```bash
uv run --no-project --with numpy --with nbformat --with requests python -B \
  -m unittest discover -s analysis/marlin-fixed-replay-20261011 -p 'test_*.py' -v
```

After freezing, live read-only admission and the authorized one-attempt launch use the installed SDK interpreter:

```bash
~/.local/share/uv/tools/kaggle/bin/python -B \
  analysis/marlin-fixed-replay-20261011/execute.py --dry
~/.local/share/uv/tools/kaggle/bin/python -B \
  analysis/marlin-fixed-replay-20261011/execute.py
```

The collector has no launch path and uses a NumPy-capable subprocess for auditing. Do not rerun execute.py after an exclusive attempt receipt exists. Resume only collect.py if collection was interrupted.

Live admission checks the reported allowance as well as used/reserved time. On 2026-10-10 the quota service reported 21600 seconds allowed but 34386 seconds already used, conflicting with the old hard-coded 108000-second cap. Until this is reconciled, the launcher preserves the quota receipt and stops before any GPU push. The browser preview was not authenticated, so it could not verify the signed-in UI quota. User authorization persists; another launch approval is not required once quota is resolved, provided this identity remains unconsumed and sealed.

Local tests cover raw numerical/timing auditing, retained timing on rejected finite outputs, corruption/incomplete evidence, the quota guard, exclusive ambiguous-push containment, and collection audit failure. They do not qualify new CUDA graph/repetition behavior or Kaggle's mounted-output path. Those remaining seams are bounded by the one-attempt run and preserved failure artifacts.
