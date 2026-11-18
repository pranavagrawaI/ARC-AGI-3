# FLARE stage-3b smoke: card (frozen before launch, 2026-10-20)

## What stage 3 showed
Kernel flare-smoke-3-20261020 ran 07:28–08:10Z. Flare verified the package signature, allocated
74.8 GiB on the GPU and stayed in "Load weights" until the 40-min health limit (INVALID, load time).
Its /proc read counter rose 4.0 GB in 2,400 s (1.7 MB/s). The binary opens the payload with O_DIRECT
("open package payload for direct I/O"), and the package was on a **gcsfuse** mount. The dataset verify
kernel (CPU, 64 MiB reads from 6 threads on one fd) has been running for more than 3.5 h, so this cold
private dataset is slow even for large reads. Notebook-output inputs are copied onto local ext4 before
the notebook starts; probe 5 read the six FLARE parts there at 504 MB/s.

## Change
`../readahead/flare_ra.so` is preloaded into Flare through the trixie loader (`--preload`). It
recognizes the package by dev/inode, drops O_DIRECT, and serves every read from a 64 MiB-block cache.
16 threads fetch blocks with fresh-handle buffered reads, 24 blocks ahead, 64 blocks resident.
- Route A: the dataset file itself.
- Route B: a sparse placeholder of the package's size; the library serves its bytes from the six
  staged parts (`FLARE_RA_SOURCES`). Flare runs with `--verify-weights`, so if a byte ever came from
  the placeholder, Flare's own check fails the load instead of serving garbage.
- A 20 s raw benchmark of the fetch pattern runs on each route first; the faster one is used.
- Local evidence: `../readahead/test_ra.py` PASS in both modes (pread from 8 threads, read, readv,
  preadv2, copy_file_range, sendfile, splice, mmap; byte-identical). The library loads under the
  shipped glibc 2.41 with the bundled flare binary, and Flare's package open is tracked. `dryrun.py`
  forces each route through the full smoke against the mock server.
- Everything else is stage 3 unchanged: flags (incl. `--prefix-cache-memory 64G`), functional rows,
  the S=10/14/16 replay, estimators, the SGLang M3A reference (U_overlap 477.6).

## Decision: SMOKE-CARD.md's rule, unchanged
INVALID if Flare is never healthy; KILL if fewer than 3/6 functional rows parse, replay errors exceed 2%,
or r < 1.10; PAIRED if 1.10 ≤ r < 1.30; GO if r ≥ 1.30. Added validity conditions for route B: the
library's stats at health must show `mmap_large` = 0 and served bytes ≥ 99 GB (all of the payload went
through it). Otherwise the run is INVALID (route leak), whatever r is.

## Reported, not decided
Load time L (Flare start to health) and the library's fetch MB/s, waits and read-size histogram; both
route benchmarks; Kaggle staging time of the parts (kernel RUNNING → notebook start). For a submission,
route B costs that staging time and route A costs its load time; r_eff per SMOKE3-CARD.md.

## Cost
One RTX PRO 6000 session: ~20 min of input staging plus at most 110 min of notebook time. Weekly GPU
quota was about 21 h left before stage 3, which used 0.7 h.
