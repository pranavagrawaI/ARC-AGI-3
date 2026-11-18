# Local reassessment: retain chunk 2048

The four completed arms can be assessed without another GPU run. This is a separate local
reassessment, not a rewrite of Kaggle's ERROR or the original failed lifecycle result.

## Repairs and checks

`deployment/kaggle/prefix-cache-20260907/reassess.py` is a successor to the frozen assessor.
It reads exactly `batch-captured-1.json`, `batch-captured-4.json`, `batch-long95000-1.json`
and `batch-long95000-4.json`. The original `batch-*.json` glob also matched request/result
files: 36 matches per arm. Regression checks use the actual retained directory, not a
summary-only fixture.

Arm 3's sole rejected entry is the exact timestamped INFO message
`Waiting for background tasks to complete. (CTRL+C to force quit)`.
The successor classifier accepts only that full message when it was previously classified
as an unmatched nonempty line. It verifies classified text against byte ranges in server.log;
all other rejected entries remain rejected. It separately requires successful containment,
quiet window, TERM-only shutdown and root return code zero. It will not excuse a different
failure classification. Negative tests cover warnings, altered text, mismatched raw bytes and
failed lifecycle gates.

All four arms pass this reviewed lifecycle gate. Only one entry changes classification.
All 602 files under the retained run root were hashed before and after; none changed.
The launched notebook, original assess.py, manifest and experiment card remain untouched.
`build.py --check` still passes. This does not install a new allowlist into a future runtime.

Evidence: `reassessment-v1/REASSESSMENT.json` contains the original lifecycle records,
reviewed classifications, source hashes, raw-file inventory, timings and natural results.

## Performance disposition

| Workload | 2048 arms, seconds | 4096 arms, seconds |
|---|---|---|
| Captured C1 | 30.092 / 30.188 | 31.129 / 31.142 |
| Captured C4 | 11.599 / 10.695 | 11.003 / 10.999 |
| Mechanical 95k C1 | 134.108 / 133.993 | 130.050 / 130.075 |
| Mechanical 95k C4 | 103.560 / 103.523 | 98.873 / 98.868 |

No candidate reaches the 10% selection bar. Captured C1 regresses; the 95k gains are small.
Captured C4's two baseline timings differ by about 8%, so that cell is not a 0.1%-repeatability
result. All finite-set requests meet the diagnostic 180-second common-ready threshold; this
does not establish real game deadlines. Natural validation retains three unknown-tool outputs:
one in arm 0, none in arm 1, one in arm 2, one in arm 3. These small samples do not measure a
quality-rate difference. Retain 2048, promotion=false, MTP remains parked.

## Correction: prefill graph settings did change

The raw `resolved-policy.json` records differ in `chunked_prefill_size` **and**
`cuda_graph_config.prefill`. The 2048 arms have prefill max_bs=2048; the 4096 arms have
max_bs=4096 and extra captured sizes 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096.
Decode graph settings and recorded Mamba settings match. The previous write-up's claim that
every graph setting was identical is contradicted by these files. The comparison remains a
chunk-configuration treatment, not isolated kernel acceleration.

## Why 2048 fewer tokens were reused

The shared prefix for normal_long_12_13 ends at token 19708. Its cold predecessor leaves
chunk-boundary recurrent-state checkpoints. The latest available checkpoint before divergence
is 18432 with 2048 chunks, versus 16384 with 4096 chunks. The short pair diverges at 9199,
so both configurations reuse the checkpoint at 8192.

The pinned wheel's code supports this mechanism:

- `schedule_batch.py:2693-2785` tracks a checkpoint for each prefill extension and can
  override it with a known aligned branch point.
- `scheduler.py:3090-3091` stashes unfinished chunked requests through cache insertion.
- `mamba_radix_cache.py:675-803` inserts the tracked Mamba state with the matching prefix.
- `mamba_radix_cache.py:1123-1125,1195` limits reusable KV to a node carrying Mamba state.
- `runtime_context.py:1633-1644` and `server_args.py:10250-10277` distinguish the model/page
  checkpoint grid from the configured prefill chunk size.

`cache_checkpoint_probe.py` executes the actual pinned checkpoint-position method on CPU
fixtures with a 64-token state grid and page size 1, then compares predictions to all three
pairs in all four arms. All 12 counts match. The retained model configuration contains no
`mamba_chunk_size` override; the pinned property documents the 64-token fallback. This is
source-supported arithmetic reproduction, not execution of GPU state copying, eviction or
the complete scheduler.

A counterexample to a universal chunk-rounding rule also passes: with prefix 16384, a 4096
extension and a known branch at 19648, the method chooses checkpoint **19648**. That is not
a multiple of 4096. Its kernel retrieval indicator is 19649; the actual stored checkpoint is
19648. Evidence and wheel/source hashes are in `reassessment-v1/CACHE-CHECKPOINT.json`.

## Next decision

Do not change the matching guard to reuse KV without corresponding recurrent state. There is
no demonstrated cache-correctness defect here. A smaller-chunk sweep is not yet justified by
the limited additional prefix work it might save versus more prefill scheduling. A useful
cheap follow-up, if needed, is whether repeated related requests acquire and reuse the finer
branch checkpoint already supported by this runtime. No new GPU run, host change, or
performance candidate is authorized or selected by this reassessment.

Verification: 25 local tests including real-artifact reassessment, negative classification
controls and pinned-method CPU reproduction; original package build check passes.
