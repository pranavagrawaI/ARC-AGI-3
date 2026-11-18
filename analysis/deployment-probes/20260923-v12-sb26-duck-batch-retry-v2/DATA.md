# Complete data index — v12 sb26 Duck-batch retry v2

This index describes the locally downloaded Kaggle output for `pranavagrawai/tycho-nooa-v12-sb26-duck-batch` version 2. All timestamps are UTC. `kernel-output/` contains the unmodified downloaded files; `EXPERIMENT-CARD.md` is the pre-run card; `RESULT.md` is the interpretation. `MANIFEST.sha256` lists each evidence file and the frozen package by SHA-256.

## Run and hardware

| Field | Value | Raw source |
|---|---:|---|
| Kaggle status | COMPLETE | Kaggle kernel v2 status, checked 2026-09-23 16:26 UTC |
| Launcher status | `passed`; return code 0 | `kernel-output/v12-sb26-duck-batch/nooa-join-public3/result.json` |
| Run clock | 07:19:35.761–08:34:15.568; 4479.807 s | `result.json` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition; 97,887 MiB | `system-before.json`, `system-after.json`; `nvidia-smi` returned 0 |
| Model | local Qwen3-8-27B FP8 via SGLang, offline | `official-launcher-result.json`, runtime provenance |
| Total chat-completion accesses | 268 successful: 267 solver, 1 warmup | `request-count-facts.json` |
| Games | sb26, two sequential passes | `official-launcher-result.json`, `nooa-join-launcher.stdout.log` |

## Episode facts

| Field | Pass 0 (`sb26-c7_k1dh4`) | Pass 1 (`sb26-wov3ulk5`) |
|---|---:|---:|
| Start (`launch.json`) | 07:31:52.894 | 08:03:47.357 |
| Host stop | wall budget; 1911.638 s | wall budget; 1828.044 s |
| Executed actions / submitted batches | 45 / 45 | 49 / 33 |
| Action types | 30 CLICK, 11 SPACEBAR, 2 UNDO, 2 RESET | 36 CLICK, 11 SPACEBAR, 1 UNDO, 1 RESET |
| Model calls | 157 actor, 0 critic | 110 actor, 0 critic |
| Prompt / completion / reasoning tokens in model-call ledger | 4,731,892 / 60,059 / 0 | 3,430,425 / 57,193 / 0 |
| Step checks | 34 unresolved; 0 mismatched | 39 unresolved; 0 mismatched |
| Transcript evictions | 41 EVICTED, 0 EVICTION_ERROR | 29 EVICTED, 0 EVICTION_ERROR |
| Multi-action batches | none | 13 length-2; 1 length-5 |
| Maximum level | 0 | 1 |
| First L1 | none | action 16, turn 12, SPACEBAR, 08:13:03.677 |
| Episode-clock first L1 | none | 556.320 s (9 min 16.320 s) |
| Final game state | NOT_FINISHED | NOT_FINISHED |

Pass 1 executed both actions in 12 of its 13 length-2 batches and all five actions in the length-5 batch. One length-2 batch stopped after the first action. Pass 0's host stop preceded pass 1's start by about 2.8 s. The pre-run prediction that **both** passes would reach L1 failed. The successful pass beat the prior critic-off 17.5 min episode but missed Duck's 322 s upper target.

For the more closely matched successful-life comparison, the critic-off run reached L1 in 15 actions and about 12.5 min after its RESET/GAME_OVER split; this run reached L1 in 16 actions and 9.27 min. The critic-off life used 67 model calls before L1 versus 46 here. See the follow-up comparison in `RESULT.md`; this is a cross-run observation, not a causal attribution.

## Assessor and limits

`ASSESSMENT.json` is `INVALID` with 20 errors: two missing per-session stop files, two expected-stop errors, and 16 `action executed after interrupted prefix` errors. The two host stop rows exist at `solver-workspaces/game-stops.jsonl`, one per pass. All step checks are `unresolved`; with `duck_batch=1`, the worker deliberately retained queued actions after these checks. The assessor's interrupted-prefix rule assumes the queue was discarded and therefore misclassifies those executions. `result.json` passed the launcher and lifecycle checks. These observations establish played actions and one L1, not a solved game or official score.

## Raw file map

All paths below are relative to `kernel-output/v12-sb26-duck-batch/nooa-join-public3/` unless specified otherwise.

- `result.json`, `official-launcher-result.json`, `request-count-facts.json`: launcher, run, and model request accounting.
- `system-before.json`, `system-after.json`: machine and GPU snapshots.
- `solver-workspaces/game-stops.jsonl`: authoritative two host stops.
- `solver-workspaces/<session>/launch.json`: factor settings, source identity, start time.
- `solver-workspaces/<session>/ipc/actions.jsonl`: submitted action lists, including batch lengths.
- `solver-workspaces/<session>/ipc/transitions.jsonl`: every executed action and before/after state.
- `solver-workspaces/<session>/ipc/model_calls.jsonl`: 267 full model request/response records.
- `solver-workspaces/<session>/ipc/step_checks.jsonl`, `transcript_evictions.jsonl`, `integrity.jsonl`, `execution_timing.jsonl`: prediction, eviction, integrity, and execution timing evidence.
- `nooa-join-launcher.stdout.log`, `.stderr.log`, `server.log`, `events.log`, `teardown.log`: runtime chronology and diagnostics.
- `kernel-output/v12-sb26-duck-batch/server_recording/*.recording.jsonl`: both host game recordings.
- `kernel-output/tycho-nooa-v12-sb26-duck-batch.log`: Kaggle notebook log.
- `kernel-output/qwen38-fp8-r8l3b-bootstrap/`: runtime installation and provenance.
- `kernel-output/nooa-join/` and `kernel-output/v12-sb26-duck-batch/nooa-join/`: emitted runtime source copies.
- `deployment/kaggle/submission-runtime-20260920-v12/` in the evidence archive: the exact frozen package used for the push, including `LAUNCH.json` and `probe.ipynb`.

The complete file-level list with hashes is `MANIFEST.sha256`. No raw file was rewritten for this index.
