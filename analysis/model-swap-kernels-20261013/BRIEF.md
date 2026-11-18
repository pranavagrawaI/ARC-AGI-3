Context (2026-10-13, ARC-AGI-3 Kaggle, deadline 2026-11-02). Plan: board thread 682. Pranav authorised (2026-10-13, "yes do it") preparing and smoke-running two model-swap candidates from analysis/model-swap-survey-20261013/shortlist.json, then queueing a passing candidate for a hidden draw. $0, no training.
Actor name: codex-model-swap. Writable dir: analysis/model-swap-kernels-20261013 (subdirs tinfield/, azfocus/). Read-only elsewhere.

Base: the incumbent ledger notebook (incumbent/stack.json -> analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb; served model = Intel/Qwen3.8-Flash-Next-W4A16-AutoRound + albucino INT4 MTP drafter on SGLang pennyroyal, Marlin W4A16; NOT NVFP4). Each candidate changes ONLY the model weights (and what the format change strictly requires). Run incumbent/check.py on each notebook; carried pieces must remain.

Candidates:
A. Tinfield-1 (badtheorylabs/Tinfield-1; Kaggle model natnitarach/tinfield-1-nvfp4-probe, NVFP4 experts + INT8/BF16 PLE). Find the lowest-risk path onto our stack: (i) serve its NVFP4 on our SGLang if supported on Blackwell, or (ii) produce a W4A16 equivalent ONLY if a ready public W4A16/GPTQ/AutoRound build exists (no local quantisation of 177B on an 8 GB laptop). Keep the base MTP drafter; record acceptance.
B. AZ-Focus (AxonZeta/Qwen3.8-Flash-Next-AZ-Focus): only some attention/shared-expert output projections differ from the base. Diff the BF16 tensors vs Qwen/Qwen3.8-Flash-Next (use HF safetensors headers / ranged downloads; do not download 350 GB locally), check which of those tensors Intel's W4A16 quantised vs kept BF16, and graft: unquantised tensors copied directly; quantised ones re-quantised with the same AutoRound/GPTQ scheme only if feasible from the BF16 slices, else stop and report. Publish the grafted checkpoint as a private-then-public Kaggle dataset/model ONLY if public availability is needed for the submission (competition rule: external models must be public and free); ask on board thread 682 before making anything public.

Per candidate, the deliverable is a SUBMITTABLE kernel whose own completed run is the smoke:
- In interactive (non-competition-rerun) mode it plays a short smoke: about 40 min of play on the public games, then completes cleanly. In competition rerun mode it must be byte-identical in behaviour to the base except the weights. Verify how the base notebook branches on rerun vs interactive and keep that branch untouched.
- Smoke pass = server READY, MTP accept >= 2.0, finite answers, 0 crashed games, and levels-at-minute-30 not far below the c14 control curve (c14 runs: 55 levels at 30 min across 25 games; base 66). Treat this as a breakage gate, not a selection gate. Record levels-over-time with analysis/pruning-transfer-20261011/readout/levels_time.py, plus tok/s, accept length and thinking tokens per action against the control logs.
- On pass: add {"<UTC date>": {"kernel": ..., "version": ..., "message": ...}} to analysis/hidden-draws-20261010/queue.json for the next free day (before 20:30Z for today), with the message citing smoke evidence. Today 10-13 is free. Do not call competitions submit yourself; the timer does it. Order of draws: whichever passes first; the second takes the next day.

GPU budget: this week's quota has ~8 h left (resets 2026-10-17 00:00Z; API timeUsed is right, totalTimeAllowed is wrong, UI 30 h is truth). Hard cap for this task: 3.5 GPU-hours total; one launch per candidate plus at most one retry for a pure infrastructure failure. Max 2 concurrent Kaggle GPU sessions account-wide (QUEUED counts) - check active kernels first. Post a GPU slot claim on board thread 682 before every push.

Known traps (from our history; read these before building):
- Kernel title must slugify to the id; slug equal to one of our dataset slugs -> 409; a kernel id whose push was refused stays poisoned -> new id, don't re-push.
- GetKernelSessionStatus with version label 404s; bind via current identity + current_version_number; download output with exact version_number. Unknown own slug -> 403. Metadata drops dataset /version.
- python -c payloads >128 KiB hit E2BIG in notebooks; test subprocess cells via real subprocesses.
- Package __pycache__ fails verify; run python -B and check for __pycache__ before push.
- New env vars may be stripped by worker env allowlists; confirm they arrive.
- Slow-disk startup: wait for READY (up to 1300 s from start), never assert a fixed deadline.
- Pruned/changed target vs full drafter expert maps broke loading once (c301ad3eb); check drafter/target compatibility.
- OAuth token expires after 12 h; refresh via KaggleCredentials.
Write FINDINGS.md + RESULT.json in each subdir; post results on board thread 682 (python3 ~/.agent-board/board.py reply ... see --help). Say plainly what failed and why.
