# Five active experts with top-ten normalization: no supported performance win

Implemented and ran one bounded screen on 2026-10-14. **The candidate decoded faster but completed far fewer levels. The comparison is formally INVALID, so it does not establish a causal quality regression or an efficacy gain. Keep the unchanged control.**

## Observed outcome

One RTX PRO6000 session, control then candidate, each with a 900-second fresh-game play horizon. The two model servers were separate processes; the gameplay harness was inadvertently reused in one Python process. Same 25 public games, Intel W4A16 weights, top10 INT4 MTP, c14 concurrency/context/drain and sampling. No training, competition submission, incumbent promotion or second GPU launch.

| Measure | Native top10 control | Active5 / reference10 candidate |
|---|---:|---:|
| Completed levels by exactly900s | 19 | 5 |
| Final shutdown snapshot levels | 20 | 5 |
| Aggregate decode, benchmark minutes2–14 | 845.66 tok/s | 1,018.55 tok/s |
| Per-stream decode, same window | 62.59 tok/s | 74.73 tok/s |
| Median MTP acceptance length | 2.59 | 2.51 |
| Median running requests | 14 | 14 |
| Retractions | 0 | 0 |
| Mean final-snapshot public score | 3.55 | 0.62 |

Observed aggregate decode ratio **1.20445**, with **14 fewer levels / −73.68% completions** at the equal900s horizon. Final scores are shutdown snapshots, not exact900s scores. The extra control completion happened during drain and is excluded from the primary level comparison.

Kernel: `pranavagrawai/expert-reference-mass-screen-20261014`, id138642158, version1, live COMPLETE. Notebook containment completed after2966.74s; last kernel-log event was2977.99s, about49.6min including startup. Both are within the authorized3600s cap. Account quota readback after collection was25.40h used /4.60h remaining; that is an account snapshot, not an isolated billing receipt.

## What was implemented and directly verified

`routing.py` selects top10 router logits, computes their fp32 softmax, then returns the five largest entries and their original expert IDs. Algebraically, the global512-way softmax denominator cancels in p_i/sum(top10 p). The surviving weights are not rescaled to sum to1. Shared expert path and draft routing remain native. All512 expert weights stay present; this targets per-token work, not a large weight-memory saving.

`patch_runtime.py` binds the exact wheel SHA `d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225` and original qwen2_moe.py SHA `441a5e992c7299a0a6be1a3268706759062f6a15bf9a0c8350e24def051c4c2f`. Control restores original wheel bytes. Candidate changes one constructor routing seam and links the same target files through a top5 config view. Installed candidate source SHA is `058f22dce20166c43343431c1f42812e8fdad00838a542ca508b0893b5f2472c`.

Runtime receipts confirm48 distinct target layers with active5/reference10 and unchanged top10 draft. Both effective configurations are14 requests,102400 context,84 Mamba slots, AutoRound W4A16/BF16 compute, FP8 KV and MTP steps3/draft4/topk1. Both full-model finite-logprob probes returned finite outputs.

Collected220 selected raw files, rehashed with zero mismatches, and bound kernel id/version before and after collection. All50 game event files have a fresh score0/level1/action0 initial receipt; all25 initial boards match across arms, and action-event completion totals reconcile with final benchmark totals. `readout.py` uses action-history cumulative wallclock plus each game start offset to enforce the900s boundary, rather than a first-transcript clock.

Local gates: ten numerical/source/containment tests, nine passed and one GPU-only test skipped locally. An additional fresh-process isolation fixture passes. Numerical tests use an independent float64 global-probability oracle, native top10 function extracted from the exact wheel, tied/extreme/empty/dtype cases, and drift rejection. The real GPU test command completed through the launcher's fail-fast path, but its detailed success stderr was not saved; see validity below. Local CPU Torch2.14.1 differs from packaged GPU Torch2.13.0+cu130, so the CPU tests are not a substitute for the missing GPU receipt.

## Why the comparison is invalid

**Harness clock reuse.** The frozen c14 harness caches `_PRIORITY_GATE` process-wide. `configure_clock` retains the earliest start and deadline. Unpickling a new benchmark did not create a fresh gate, so the candidate inherited the control's expired clock. The exact frozen class replay in `harness_audit.py` reproduces it: first phase[100,1000], second[1500,2400], reused clock remains[100,1000]; a fresh gate correctly takes[1500,2400].

Full request snapshots contain13 prior-prefix loss/change events in13 control games and11 in11 candidate games. These include genuine large message drops. They establish exposure to the handover path, whose queue refresh uses the gate clock. The magnitude of the ranking/completion effect is unmeasured: we did not capture gate snapshots/admission decisions. Do not claim this bug explains the observed14-level loss, and do not attribute all that loss to omitted experts.

**GPU qualification receipt gap.** The inherited `run()` helper prints successful stdout but suppresses successful stderr. unittest writes its detailed results to stderr. Its gated command completed, but the saved output cannot directly establish all seven test results or graph-test skip status. The explicit graph-qualification receipt required by the card is therefore missing. This is an observability defect, not evidence that the numerical test failed.

`RESULT.json` preserves the observed metrics and marks the run INVALID with these two reasons. The frozen card, candidate and budget stayed unchanged after launch. The post-run audit makes the observed comparability and receipt failures explicit; no GPU retry was performed.

## Diagnostic signal, with attribution limits

Across completed response receipts including shutdown drain:

| Measure | Control | Candidate |
|---|---:|---:|
| Requests / completed responses | 470 /454 | 301 /282 |
| Completed response tokens | 712,071 | 875,425 |
| Output-length cutoffs | 2 | 12 |
| Mean tokens per completed reply | 1,568 | 3,104 |

The candidate produced almost twice as many tokens per reply and substantially fewer replies, despite faster decode. That is a concrete symptom of poorer conversion of compute into interactions/progress. The shared-clock confound and changed trajectories prevent a clean attribution to expert capacity versus normalization or scheduling. This screen provides no support for retaining active5/reference10.

## Repair and next decision

`repaired/` is **CPU-prepared and unlaunched**. Each gameplay arm runs through `isolated_phase.py` in a fresh interpreter, verifies a fresh gate before play and writes a separate PID/gate receipt. Parent containment owns the model-server and gameplay process groups. GPU unit stdout/stderr are saved, and an explicit changed-input CUDA graph success marker is required before model loading. Notebook schema/embedded-source compilation, asset hashes and incumbent checks pass; the process-isolation fixture directly exercises two distinct child interpreters. Full Kaggle runtime remains untested.

A clean, separately authorized comparison is needed before a causal quality or efficacy judgment. First fix control isolation and qualification evidence; do not spend a new run changing expert counts or normalization parameters while leaving the comparison broken. The incumbent and original frozen v1 files remain preserved.

## Collection and reproducibility

Initial file downloads hit intermittent output-host DNS/read failures; a parallel per-file signing path then hit API429. Collection was completed using the signed CDN links already returned by `ListKernelSessionOutput`, with metadata id/version binding, TLS validation, atomic files and resumable hashes. The current-session link method is recorded separately from explicit-version download RPCs. Movies, prompts and solver_analysis were excluded from the220-file receipt scope; their remote outputs remain available. Raw local output stays excluded from Git/T3 checkpoints; hashes, source and reports are tracked.

Reproduce the raw audits:

```bash
python3 -B analysis/expert-reference-mass-20261014/readout.py
python3 -B analysis/expert-reference-mass-20261014/harness_audit.py
python3 -B analysis/expert-reference-mass-20261014/test_isolation.py
```

The frozen card is `EXPERIMENT-CARD.md`; source/metadata/cap identity is `launch-identity.json`; collection custody is `collection.json`; raw data is under `output/reference-mass/`. The local numerical gate needs CPU Torch and `REFERENCE_WHEEL` pointing to the exact archived wheel. No command above launches inference or submits to the competition.
