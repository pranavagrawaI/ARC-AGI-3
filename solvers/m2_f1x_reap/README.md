# M2 F1x + REAP448: one stack carrying every retained piece

F1x and REAP448 + c18 were built as separate lineages, and `incumbent/check.py` fails both. F1x is the harness: B1, stdlib preload, and a 16K output cap. It was drawn as D10 and scored 30.88. REAP is the serving stack: a pruned target, a full drafter, and a fail-fast startup guard. It was drawn as D11. This tree merges them. `build.py` takes every cell of F1x v1 and swaps in REAP v1's serving cell at F1x's context. Plan approved by Pranav on 2026-10-20.

| Change from REAP's serving cell | Why |
|---|---|
| CTX (80+12+8)K → (116+16+8)K | F1x's window: 116K input plus 16K output. The B1 harness has never been drawn alone at a shorter context. B1 + c14 at 92K was our lowest draw (23.56). |
| 18 → 14 streams (MAXREQ, CUDA graph max, 84 Mamba slots) | Keeps KV pressure at F1x's own level. See below. |
| Late-READY limit: screens only | REAP v1 dies if the server isn't READY by 1,300 s. In competition mode that turns a slow disk (1 of 32 kernels loaded weights for over 1,200 s) into a zero. A submission now keeps waiting while the server process lives. A dead server still ends the run (fail-fast). |

**Sizing.** F1x on the unpruned pool (1,012,736 tokens, 10 streams) loads KV at p50 0.62 and p90 0.81, which is about 82K tokens per stream at p90. REAP448 frees about 51% more pool: 1,373,568 tokens at 108 Mamba slots, with each slot costing about 4.2K tokens. At 84 slots, expect about 1.47M. Then 0.81 × 1.47M / 82K ≈ 14.4 streams. Measured aggregate decode: 697 tok/s at 10 streams (F1x), 830 at 14 (c14), 921 at 18 (REAP c18).

`incumbent/check.py` passes on the merge with no `--without`.

## Verify

```bash
python3 -B solvers/m2_f1x_reap/build.py
python3 -B -m unittest solvers/m2_f1x_reap/tests/test_build.py solvers/m2_f1x_reap/tests/test_gates.py
python3 -B -m unittest analysis/hidden-draws-20261010/test_requires.py
```

`test_build.py` proves three things:
- The merge minus the serving change is F1x.
- Its launcher is REAP's launcher plus exactly the listed edits.
- Removing F1x's harness cells recovers REAP, except the screen soft end.

It also runs the real cell code for the branches a public screen can't reach: no soft end when TRUE_SUBMISSION, and the ready-wait loop with a live, slow server in both modes. `test_gates.py` runs the analyzer and queue step over synthetic passing and failing outputs built from real runs.

## Screen and draw

`experiment/screen-1` holds one merge arm, 60 minutes, read against the B1-lineage screens (76, 72, 65). The card is in that directory. `run_screen.sh` launches it once, collects, analyzes, and on PASS queues the exact screened v1 for the first free day from 2026-10-22, with `requires: [D11, D12]`. `daily_draw.py` submits it only after REAP's first hidden run and F1x's second have both scored above 0. If either failed, the day falls back to F1x. While either is pending, the draw waits until 10:00Z.
