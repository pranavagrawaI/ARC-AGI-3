# V16 preflight (pre-push)

## Frozen candidate hashes

```
0040f7cd9a6cedbd173a66166d885f0b5ef4beae703d0914ebeb1d5212a8d1eb  exact_request_outer_runtime.py
fd4031afec63975cd5bc98e8cc32445f54b510ac927aa486708447ca02b2d21d  frames.json
b9bba3cedc1939a0982e974bc244c63f9b610c7ef99e9e35b05ec4eb000ebb91  s0_vision_readback.ipynb
a52cdbbde7b2870205afdcf6ba2a72ab0f27899335b4877cd76ad2cdb4e5afca  vision_readback_probe.py
```

## Local verification

- Candidate tests: **41 passed** (`deployment/kaggle/s0-vision-readback-v16-candidate/tests/`).
- Ruff clean on `vision_readback_probe.py` and `build_notebook.py`.
- Notebook builder validates its own output: four cells, all code cells compile, the asset cell
  writes all three V16 assets, the run cell keeps `--r8-l3b` / `--vision-readback-probe` /
  `KAGGLE_IS_COMPETITION_RERUN`, and the install cell carries no Solver wheelhouse.
- Card/candidate agreement checked programmatically: frame hash matches the card, no competition
  source attached, GPU on and internet off, `NvidiaRtxPro6000` pinned, no `submission.parquet`,
  no `TYCHO_PUBLIC_GAME_IDS`, probe request count is 52.

## What the tests actually pin

- `grid_text` is **byte-identical** to the reference `tycho/workspace/workspace.py` `spaced_grid8`,
  so the text arm measures the incumbent input rather than a lookalike.
- Every PNG scale round-trips through Pillow to the exact `PALETTE_16` colour at each sampled
  cell's centre pixel; the scale-32 render is 2048x2048 = 4,194,304 px.
- Ground truth is recomputed from the frame arrays, never from a model answer. A regression test
  feeds frame 0's answers against frames 3 and 5 and asserts the score **falls**.
- The cell sample is spread across ≥ 32 rows and no single colour covers more than half of it, so
  the chance floor is 0.25–0.33 rather than "guess the dominant colour".
- A missing or malformed tool call scores zero, not `None`.
- A simulated downscaling server is reported as `one_token_per_cell: false` rather than passing.
- The runtime mode refuses to run without `--r8-l3b`, alongside `--official-launcher-probe`, or on
  a non-Triton backend, and aborts when the probe's request count does not reconcile with SGLang's
  access log.

## Budget

| | value |
|---|---|
| chat completions | 53 (1 runtime warmup + 48 readback + 4 token measurement) |
| environment actions | 0 |
| startup timeout | 1,200 s |
| probe timeout | 5,400 s |
| scale-32 payload | 40,696-byte PNG, 54,286-char data URL |
| text arm payload | 9,555 chars (frame 0) |

## Not verified locally, and why

The model is a 27B FP8 checkpoint on a 96 GB GPU; this machine has an 8 GB laptop card, so no arm
of the probe can be exercised against the real model before the run. Every check above is
therefore structural. The two numbers the card turns on — the measured visual-token cost of the
scale-32 render, and readback accuracy against the greyscale-heavy ls20 palette — are only
obtainable on the Kaggle route.

## Status (kernel 2)

Frozen and ready to push. Not pushed.

## Kernel 1 defect and its regression test

Kernel 1 died in 62 s at argument parsing (`--request-timeout-s 600` against the R8-L3B pin of
180); see `TERMINAL-RESULT-kernel1.md`. The probe now takes its own `--vision-request-timeout-s`
(600), and two tests parse the **built notebook's** command list through `parse_args`.
Reintroducing the defect fails exactly those two tests with the exact Kaggle error message.
