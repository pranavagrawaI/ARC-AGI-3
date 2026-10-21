# F1x hidden-run zeros: independent diagnosis, 2026-10-21

## Coordinator correction, 2026-10-21

The initial recommendations below are historical and superseded where noted here. Hidden-run root cause remains unresolved. A dead server is a demonstrated source path, not an established explanation of D9/D12. The approximate 3.5-hour dead-endpoint duration is a timing estimate under tested settings, not a completed production-timing result: `out-dead-prod`, `out-dead-fast`, `out-500-prod`, and `out-400-prod` were still running under `/tmp/f1x-faults` when checked. Completed shortened fake-server evidence is preserved in `faults/historical-cpu-snapshot-20261021/`, explicitly as historical pre-review runs with unavailable executed-notebook hashes.

Do not use the time-coded ERROR proposal below. The current guard design uses ordinary lifecycle logs, bounded process-group cleanup/restart, and an end guard; no deliberate delay to encode a failure. Independent review found that the first end guard ignored generated tokens in action history and could falsely reject a played zero. Corrected source and regression tests require separate review; old CPU runs do not qualify the corrected build. A live process rejecting requests remains outside process-exit recovery.

The scheduling hazard is now contained by exact-version quarantine in `analysis/hidden-draws-20261010/quarantine.json` (local commit `9895800a`): F1 v1 and F1x v1 are rejected before SDK/network access, including scheduler fallback. Twenty-four tests and an actual local CLI exit 5 verified containment. No queue or timer change was needed, and an unused daily slot is allowed. No guarded hidden submission or promotion is authorized by these diagnostics.

Scope: read-only remote inspection plus local CPU checks. No GPU launch, submission, upload, or queue/timer change.
Every claim below is labelled SUPPORTED (evidence in this directory), EXCLUDED (evidence against), or HYPOTHESIS.

## Symptom

| draw | ref | kernel / scriptVersionId | status | public | totalBytes | wall clock |
|---|---|---|---|---|---|---|
| D9 F1 | 57342990 | m2-f1-screen1-f1 / 359252919 | COMPLETE | 0.00 | **3411** | ≤ ~2.4 h (COMPLETE when first checked) |
| D10 F1x | 57385881 | m2-f1-screen2-f1x / **359339222** | COMPLETE | 30.88 | 3822 | not recorded |
| D12 F1x | 57477659 | m2-f1-screen2-f1x / **359339222** | COMPLETE | 0.00 | **3411** | ≤ 2.2 h (10:31Z submit, COMPLETE at 12:44Z) |

All 17 scored submissions are 3633–3900 B, and size rises with score (0.67 → 3635, ~1.5 → ~3640, ~3 → ~3840, 23–31 → 3780–3900).
The only two files below that range are the two zeros, and they are byte-count identical (`submissions-listing.json`).

## Supported

1. **D12 and D10 ran the same code.** Both bind scriptVersionId 359339222. The zero is therefore a run-to-run event, not a code
   difference between those two draws.
2. **The zero runs ended early.** Healthy hidden runs take about 9 h (the 532-min per-game cap). Both zeros were COMPLETE within ~2.2–2.4 h.
   So the games ended through an early-termination path, not by playing badly until the cap.
3. **Identical 3411 B for both zeros** points to the same degenerate output: an all-games-zero scorecard, so the result content does not depend
   on what was played. This is not proof, because the gateway schema is unknown (a local pyarrow simulation shows all-zero scores make
   the smallest file, `pqsim` in the transcript). It is not the offline-mode dummy: the notebook's 1-row dummy is 2648 B in all 29 local copies.
4. **A known silent-zero path exists, and it is shared with the control notebook.** In the server cell, `if proc.poll() is not None:
   show_log_tail(); break` does not raise. A dead or never-ready SGLang lets the benchmark run against nothing. The solver then gives
   up each game after 11 consecutive analyzer failures (`solver.py` MAX_CONSECUTIVE_ANALYZER_FAILURES=10, first request has a 900 s
   grace), so the run ends early with COMPLETE and zero levels. dfranzen-m2-raw (control) has the same lines. Only the REAP notebook (D11)
   added `raise RuntimeError("SGLang server exited during startup")`.
5. **F1x = D5 (B1, scored 26.96) plus exactly two deltas** (`diff-d5-b1-vs-f1x.txt`): (a) the sandbox stdlib preload cell, and (b) client
   context 132K / output 16K / server CTX 140K. The production config cell, run cell (TRUE_SUBMISSION branch), overlay, and server flags are
   otherwise byte-identical. So save_request_logs=False, minimal diagnostics, concurrency 120 and 532 min are all shared with a draw that scored.

## Excluded (cheap probes)

- **Preload breaks the sandbox.** The extracted B1 sandbox matched the notebook's hash (60de76…), the F1 patch reproduced the
  notebook's hash (e8aca9…), and it ran 2,400 calls at 120-way concurrency with 0 errors, p50 0.50 s, max 0.95 s against a 60 s
  limit (`sandbox-preload-stress.json`). It is also timing-inconsistent: a broken tool sandbox leaves the analyzer answering, so games
  would run to the 9 h cap, not end in ~2 h.
- **140K context pushes the server over a memory edge.** Screen serve.logs give F1x 4.23 GB free after graph capture vs B1 4.25 GB,
  and a KV pool of 1,012,736 vs 1,011,264 tokens. No serve.log (F1, F1x, B1 ×4) has a traceback, OOM, or context error.
- **F1x starts slower.** READY took 469–589 s in every arm, F1x being the fastest at 469 s. The warmup RESET fires within 1 s of the
  benchmark start in all arms.
- **Early scorecard close race.** `Benchmark.run` starts all 110 games one after another (`open_run` reaches 110) before any game
  plays. A start failure propagates and raises, which gives ERROR, not COMPLETE-0.
- **Our own concurrent GPU sessions / T4 placement caused by them.** No own kernel ran during D12 (last own run was 10-20 15:56Z).
  D11 overlapped five of our own sessions and scored 23.32.
- **Quota as a duration channel.** GPU timeUsed did not change across D10's 9 h rerun, so reruns are not billed to us.

## Hypotheses still alive (ranked)

- **H1, most supported: the SGLang server was dead or never usable in the hidden rerun** (crash at load, wrong or degraded GPU,
  slow/failed checkpoint read, driver fault). The notebook does not raise on this, the solver quits each game after 11 failures, and
  the result is an early COMPLETE with an all-zero card. This fits all three signatures (short run, zero, identical bytes). Nothing
  in it is F1-specific.
- **H2: a transient gateway- or infrastructure-side failure** that zeroes every game (for example the gateway refusing all actions).
  The early end and the identical bytes fit H2 as well. It cannot be told apart from H1 without in-run evidence.
- **H3: an F1-family-specific cause.** Two of three F1-family hidden runs scored zero against 0 of 10 other draws (Fisher one-sided
  p ≈ 0.04, and the grouping was chosen after the fact). No mechanism in the two code deltas survives the checks above. Keep H3 open
  only as a rate signal.

## Recoverability of the exact hidden evidence

- `GetKernelSessionLogsStream` works by kernel_session_id: it returned the original screen log for 359339222
  (`logstream-359339222.bin`). No API lists competition-rerun session ids, and `get_submission` exposes only ref, bytes, date,
  description, errorDescription, scores, status and url. Episodes are []. I did not guess session ids: they are global and sequential,
  and probing them would touch other users' sessions.
- Kaggle almost certainly withholds rerun logs for scored code submissions so hidden content cannot leak. Treat "the UI has the log"
  as unverified until someone sees it.

## Assessment of the prepared probe (f1x-production-probe-20261018)

It cannot discriminate H1 from H2, and D10 already answered its main question: the same F1x v1 ran in real production mode on the real
gateway and scored 30.88. A pass would only lower the estimated rate of an F1x-code-path intermittent failure (H3). If run at all, add:
GPU-name assert (RTX PRO 6000) before serving; a lifecycle log of READY/DEADLINE/DIED with exit code and seconds from notebook start;
`raise` on startup death; a whole-run server watchdog (proc.poll + /health every 30 s, logged with times); first successful completion,
first model-issued action and first level, each timestamped; per-sample counts of `analyzer failed` notes; and the close_scorecard result.

## Smallest discriminating next experiment

The failure happens only in the hidden environment, and the only signals out of a rerun are status, score and completion time.
So instrument the submitted notebook rather than an emulation:

- **Guards that turn each silent-zero path into ERROR, each at a coded time** (sleep before raising, poll status every 5 min):
  - wrong GPU: ~0 min
  - server died during startup: ~15 min
  - no successful completion by 30 min: ~45 min
  - server died mid-run (watchdog `os._exit`): ~75 min
  - all games ended with analyzer-failure or crash notes and zero levels: end of run plus 30 min
- **Healthy-path behaviour is unchanged**, so a healthy run still returns a valid score, and an ERROR rerun did not consume a slot on 09-17.
  If the run still comes back COMPLETE-0 with none of the guards firing, H1 is excluded and H2 (gateway-side) becomes the lead.
- This needs a hidden submission and packaging, which needs Pranav's authorization. **Arm choice is a judgement call:** guarded F1x gives
  the best chance of a recurrence (diagnosis); guarded control is the score-safe option.

## Scheduling hazard (not changed)

queue.json 2026-10-22 = m2-f1x-reap14-screen1 with requires [D11, D12]. D12 = 0.00, so submit_draw.py returns REQUIRES_FAILED, and
daily_draw.py then submits FALLBACK = **m2-f1-screen2-f1x v1**, the kernel that just scored 0. arc-hidden-draw-daily.timer is
active. The first 10-22 UTC tick (~00:30Z) will submit it unless the fallback is changed or the timer paused.
Control dfranzen-m2-raw v1 is 5/5 positive (mean 27.8), but it carries the same silent-server path.
