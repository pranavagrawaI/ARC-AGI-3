# Pruning-transfer pair readout (2026-10-12)

Kernels: probe2 138294323 (passed), control 138302697 (c14, COMPLETE, 103 levels), candidate 138302732 (REAP448 + 18 streams, ERROR at startup, 0 games).
Verdict recorded by `execute.py`: `INVALID_RETAIN_CONTROL`. The pair compared nothing.

## 1. The candidate died from our own startup check, not from pruning (`startup.png`)

- Across 32 dfranzen-family kernels, READY time is set by how fast the machine reads the 185 GB input precache: weight load ≈ precache + 30 s. Six of 32 (19%) were still loading at the notebook's 722 s deadline.
- Today's three kernels drew precache times of 232 s, 364 s and 504 s. The candidate drew the slow disk: weight load 521 s, then CUDA-graph capture, still loading at 722 s. It sits on the same line as every unpruned kernel, so pruning did not slow loading. Probe2, also pruned, loaded in 266 s.
- Stock dfranzen tolerates a miss: it releases the benchmark and keeps loading. Three of the five earlier deadline kernels went on to play. `build.py` appends a health assertion after the launcher (`urlopen(url)` must return 200). That assertion turns a slow disk into a fatal ERROR. It is a confound we introduced, and it would kill any arm, the control included, that draws a disk slower than about 0.4 GB/s.

## 2. The REAP mechanism itself held

- Probe2: target 448 experts with 193,536 / 27,648 / 48 tensors kept / dropped / routers; drafter-map guard installed; MTP acceptance 3.10; 32/32 correct answers at 14 and 18 concurrency.
- KV pool: 1,373,568 tokens pruned vs 911,424 control (+51%) at the same memory fraction.
- No gameplay evidence on levels yet.

## 3. A second c14 run, and what c14 actually does (`tempo.png`, `totals.png`)

- c14 replicate: 103 levels, mean score 44.10, 829 tok/s aggregate (E2: 107, 45.77, 815). Throughput replicates. Levels sit inside the replicate noise.
- Identical-config pairs differ by 4 (c14), 11 (base) and 15 (B1) levels, so the single-run sd is about 9. Arm means: base 101.5, c14 105, B1 109.5, B1+c14 108 (n=1). None is separated from the others by more than one replicate gap.
- Mechanism: all 25 games start together and the gate lets 10 (or 14) decode at once. c14 raises aggregate decode but lowers per-stream decode from about 74 to about 61 tok/s (−17%). All three 14-stream runs fall 12–18 levels behind the base mean by minute 15–20, then catch up and pass it between minute 50 and 100. Final margin over base: +1.5 to +6.5.
- Reading: c14 buys breadth at the cost of per-game speed. Over a 121-minute horizon the two roughly cancel, with a small late gain. If the hidden run's per-game horizon is shorter, c14 could lose. This is an observation over three runs, not a tested mechanism.

## Next checks

- Rerun the candidate alone against the two c14 runs we now have (103, 107), after replacing the health assertion with stock deadline behaviour. Wait for READY past the deadline, as dfranzen does, and record the startup time.
- Confirm the hidden run's per-game horizon before leaning on c14's late gain.

Scripts: `startup.py`, `levels_time.py`, `serve_series.py`, `figures.py`. Data: `*.json` here.
