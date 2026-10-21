# m2_guard fault screen 1 (public, interactive), frozen 2026-10-21

Status: PREPARED, NOT LAUNCHED. The GPU launch needs coordinator approval (thread d74e17d0) and must not overlap any other
of our GPU sessions.

Question: on a real Kaggle RTX PRO 6000 container, does the guarded F1x notebook:
- pass its GPU check,
- recover from a crash of the SGLang parent process mid-run (orphaned children killed, server relaunched, READY again), and
- keep its games playing through the outage, with none quitting on analyzer failures and no end-guard false positive?

This qualifies an operation. It makes no claim about gameplay or efficacy.

Candidate: `pranavagrawai/m2-guard-fault-screen1`, version 1.
- Notebook `kernel/m2-guard-fault-screen1.ipynb`, sha256 `94b2f62137d72bae944835d1d8667c1981faa88e16c5cc3cf42187c2ce693ed6`.
- Metadata sha256 `71eef644b5fa085b9fbea0ed4ef27c1a45498d28728c2cb673bc821c443b8aec`. It is the parent's metadata with a new id and
  title (`machine_shape` NvidiaRtxPro6000, same data and model sources).
- Built by `solvers/m2_guard/build.py`, which refuses to rebuild once `launch-attempt.json` exists here.
- Parent: F1x screen-2 v1, `826dfdc2…7df9` (hidden D10 30.88, D12 0.00).

The only difference from the parent is four guard edits; `tests/test_guard.py` checks this.
1. GPU check in the server cell: every `nvidia-smi` GPU must be an RTX PRO 6000, else raise.
2. Supervisor cell before the solver import: sets `ARC3_MAX_ANALYZER_FAILURES` = 40 (was 10), then polls the server every 10 s.
   On exit it runs killpg, sleeps 10 s and relaunches with the same args, at most 2 restarts. Events go to `guard-lifecycle.jsonl`.
3. Screen-only fault cell, inert when TRUE_SUBMISSION: 1200 s after READY, SIGKILL the SGLang parent alone.
4. End guard after `bm.run`: raise if there are 0 levels and either 0 generated tokens or ≥90 % of served games ended on
   analyzer failures or crashes.

Prediction:
- The GPU line names an RTX PRO 6000.
- `guard-lifecycle.jsonl` shows supervisor_start → server_ready (restarts 0) → fault_injected_kill → server_exit
  → server_relaunch (restarts 1) → server_ready (restarts 1).
- The second READY comes ≤ 900 s after the kill. The first load took 469–589 s, but the relaunch rereads weights:
  DROP_CACHE_AFTER_LOAD empties the page cache, and the precache thread does not rerun.
- No game ends with an `analyzer failed` note. The end guard prints and does not raise. The notebook completes normally.

Gates (all must hold for PASS):
- G1: the GPU guard line shows RTX PRO 6000.
- G2: the lifecycle sequence above is present, with restarts = 1 and no `server_abandoned`.
- G3: relaunch READY − kill ≤ 900 s.
- G4: no game note starts with `analyzer failed` or `error:`, and no game is crashed.
- G5: requests are completed after the relaunch READY in ≥ 5 distinct games, from the request logs (save_request_logs is on in
  screens).
- G6: `M2 GUARD end` is printed and the notebook completes without an exception.

Levels at the 60-min soft end are recorded but not gated. The injected outage costs play, so they are not comparable with the F1x pool.

Cap: one launch, one RTX PRO 6000, ≤ 5,400 GPU-seconds including startup. The expected run is about 80 min: ~10 min load,
then the 60-min screen soft end (which starts at the run cell), then teardown. No retries and no hidden submission.

Stop: normal completion, the platform hard cap, a guard raise, or a setup failure. Do not relaunch on failure. Preserve all output.

Invalid if:
- source drift from the hashes above,
- a GPU other than RTX PRO 6000 (G1 fails: the guard worked, but the relaunch question is unanswered),
- missing `guard-lifecycle.jsonl` or `serve.log`,
- the fault never fired (the server never reached READY).

Result map:
- PASS: the supervisor's cleanup and relaunch work on real hardware. The guard is eligible for independent review, then
  coordinator-scheduled use.
- G3 fails (slow relaunch): patience 40 (~17 min of failed turns) may be too short. Measure the relaunch time, then decide
  between more patience and a faster reload, before any submission use.
- G2/G4 fail with orphan or VRAM errors in serve.log: the cleanup is wrong. Fix it before any use.
- G6 raises: the end-guard rule is wrong on real data. Fix it before any use.

Already established on CPU, so not asked here (evidence in analysis/f1x-hidden-zero-diagnosis-20261021/faults/):
- the dead-endpoint silent zero,
- mid-run crash recovery against a fake server,
- startup death → abandon → end guard raise,
- healthy-path parity,
- unit tests for the GPU, end and fault cells.

Limits:
- One crash at one time, and an interactive screen, not the hidden gateway.
- It cannot show that hidden zeros were server crashes. CPU timing suggests a dead-from-start server would take ~3.5 h,
  longer than D12's ≤ 2.2 h, so a crash-from-start is not the hidden cause. A mid-run crash, or a live server rejecting
  requests, is still possible.
