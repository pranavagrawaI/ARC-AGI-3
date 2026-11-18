# Marlin v3 numerical diagnosis

The saved FP32 reference is consistent with independent CPU FP64 evaluation of all 40 rows. The shared worst error on the changed input is reproduced by the kernel's BF16 dtype boundaries. This supports retaining the reduction candidate for further investigation, while preserving v3's numerical rejection and its lack of timings.

The completed input is `marlin-reduction-replay-v3-20261010`, kernel 138024107 v1. Its original capture, card, tolerances and readout are untouched. No GPU was used for this diagnosis.

## Reproduce

Uses NumPy only. The restricted archive reader accepts this capture's uncompressed CPU tensor format; it does not execute arbitrary pickle globals. BF16/F32 decoding is checked against the separately exported saved NPZ outputs, and BF16 nearest-even conversion is checked with known halfway values.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python -B \
  analysis/marlin-numerics-diagnosis-20261010/probe.py --all-rows \
  --out analysis/marlin-numerics-diagnosis-20261010/full-probe.json
```

`full-probe.json` and its NPZ arrays retain all-row reference comparisons, cumulative dtype probes and per-expert contributions at column 268. `probe.json` preserves the first three-row resolving probe. `drift.json` and `model-comparison.json` retain the separate GPU arm comparisons. `provenance.json` binds inputs, scripts and exact-wheel source excerpts to their hashes.

## Findings

CPU FP64 agrees with the saved GPU FP32 reference within 5.09e-8 on the original input and 1.23e-8 on the changed input, across 102,400 elements each. This rules out a large error in the tested reference computation as the explanation for the 5.35e-4 shared error. Both calculations use the independently unpacked/dequantized captured weights, whose layout sentinel passed during v3.

At changed-input token 18, output 268:

| Arithmetic | Predicted output |
|---|---:|
| Independent FP64 math | -0.02543759078 |
| Saved GPU FP32 reference | -0.02543757856 |
| FP32 with final BF16 rounding only | -0.025390625 |
| Add BF16 dequantized weights | -0.02551269531 |
| Also round first GEMM output | -0.02502441406 |
| Also round gated activation | -0.02502441406 |
| Also round second GEMM output | -0.02490234375 |
| Also round route weight and product | -0.02490234375 |
| Saved control and treatment | -0.02490234375 |

This cumulative probe reproduces the shared maximum error exactly at the failing coordinate. Expert 72 contributes almost all of that discrepancy. Its routed contribution changes from -0.02718162599 in FP64 math to -0.026611328125 in the full BF16 model. The first GEMM storage rounding produces the largest cumulative change in that expert's contribution. These are model interventions; raw GPU intermediate buffers were not captured.

The relevant dtype boundaries are present in the exact wheel. Grouped quantization scales multiply dequantized values in BF16; both GEMMs store BF16; the gated activation is stored in BF16; the route weight is cast to BF16 and multiplied in BF16. The CPU model assumes FP32 final slot accumulation followed by BF16 storage. It does not reproduce CUDA scheduling, partial sums or atomic accumulation order.

The full dtype model agrees exactly with 98.39% of treatment outputs on original input and 98.08% on changed input, compared with 63.58% and 52.39% for the control. Relative to this model, changed-input normalized RMS residual is 0.000395 for treatment and 0.003245 for control. This supports a rounding explanation and is consistent with the non-atomic FP32 scratch path more closely following a single final rounding. It does not prove every residual comes from atomic ordering.

The cross-arm drift is a separate issue. On the changed input, 48,719 of 102,400 values differ; seven exceed the predeclared maximum-drift tolerance. The largest difference is 0.0001220703125, one BF16 step at token 22/output 268. There, the treatment equals the dtype model and the control differs by one step. Normalizing by the small changed-reference RMS of 0.001066297 turns that difference into 0.11448, above the 0.05 gate. The changed-reference RMS is about 21 times smaller than the original. A one-step maximum does not establish that every element differs by at most one step or that downstream token behavior is equivalent.

## Decision

There is no reference arithmetic bug demonstrated here. The current screen rejects expected production BF16 arithmetic against ideal unrounded FP32 on this changed input, and separately rejects measured control-treatment drift. Replacing the reference alone cannot make the cross-arm drift pass.

Do not retroactively change v3's verdict or claim a speedup. Its timings are absent and the remaining eight cases are untested. Do not copy this dtype model into an acceptance oracle without independent qualification.

The next useful GPU experiment is a fresh fixed-capture replay, without loading the full server again: include repeated control-control evaluations to measure the control's own variation, retain FP32 and dtype-aware diagnostic comparisons for every fixed case, and collect all numerical evidence plus diagnostic timings without promoting rejected cases. Before launch, precommit the acceptance contract, caps and a fresh identity. Timing a numerically rejected case can inform whether this direction is worth pursuing, but cannot qualify it for serving. Existing v3 launch authority is consumed.
