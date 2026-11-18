# FLARE stage-3c smoke: card (frozen before launch, 2026-10-20)

## What 3b showed (flare-smoke-3b-20261020, notebook 11:23–11:27Z)
Staging the six parts took 18.5 min (RUNNING 11:04:57 → notebook start 11:23:25). Route benchmark:
the gcsfuse dataset at 136 MB/s vs the staged parts at 873 MB/s, so route B was used. Flare checked
every stored byte with `--verify-weights` in 55.5 s (OK), which proves flare_ra.so served the right bytes.
Weights loaded (72.4 GiB in 5.3 s) and 2,481 kernels were prepared. Then "Prepare execution" failed with
`cuGraphKernelNodeSetAttribute failed (1, CUDA_ERROR_INVALID_VALUE)`, the same with and without the prefix
cache. The package (HF gb202 rev 431b3021, unchanged since 10-18) holds captured graphs built for a newer
driver than Kaggle's 580 (CUDA 13.0).

## Change (one)
flare_ra.so now also overrides dlsym; Flare resolves CUDA through dlopen("libcuda.so.1") + dlsym.
For `cuGraphKernelNodeSetAttribute` it returns a wrapper that logs every attribute id (calls, failures,
the first failing value) to flare_cu.log. With FLARE_CU_SKIP=all it reports success for any attribute
the driver rejects. Tested locally against a fake libcuda (log-only, skip-all, skip-list, inert); the
IO tests still pass, and the library loads with the bundled flare under glibc 2.41. 3b is otherwise
unchanged.

## Decision
SMOKE-CARD.md's rule (INVALID / KILL / PAIRED 1.10–1.30 / GO ≥ 1.30 on S=10 U_overlap vs 477.6), with
SMOKE3B-CARD.md's route-B validity checks, plus:
- Name the skipped attribute id(s) from flare_cu.log. Anything measured is conditional on that skip.
- If execution still fails, or the functional rows don't parse, the result is INVALID (the skip broke
  something), not KILL. The next step is to look at that attribute, not to judge Flare.
- If it runs, the attribute is a performance hint or not needed, and r is read as usual. A GO then needs a
  later check that the skip doesn't cost quality, which the 60-min F1x-vs-F1x+Flare screen does anyway.
