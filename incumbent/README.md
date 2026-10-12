# Incumbent stack

`stack.json` lists every piece of the current submission stack as a delta on the dfranzen M2 base: what it changes, the evidence it earned, and the strings that prove it is present. `check.py` fails any arm that silently drops a retained piece.

Why it exists: REAP448 (Arm P, 54.5 vs 43 levels) disappeared in the SGLang migration and nobody noticed for 11 days.

## Admission rule (2026-10-12)

We no longer expect one big lever, so small wins go in. One hidden draw resolves only about ±25%, and a public-25 pass about ±11 levels. Neither can confirm a 5% gain or see a 5% harm. So:

1. **Enter on mechanism.** The piece does what it claims, measured directly and cheaply: tok/s, a pin that fires, a guard that triggers, fewer repeated failed attempts. That earns `probation`.
2. **No-harm floor.** One public-25 pass with no crash and levels not below the control's noise band. Pass both and the piece is `retained`.
3. **Harm is judged on the bundle.** Alternate hidden draws, bundle vs unchanged control. If the bundle stops beating the control across draws, ablate the newest `probation` pieces first.
4. **Removal is explicit.** Move the piece to `removed` with its reason. An arm that leaves out a retained piece passes `--without ID`, and its card says why.

## Use

```bash
python3 incumbent/check.py ARM.ipynb --without reap448            # before push
python3 incumbent/check.py ARM.ipynb --logs ARM/output            # after the run: markers fired
```

Update `stack.json` in the same commit as the decision that changes it.
