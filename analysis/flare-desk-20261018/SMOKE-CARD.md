# FLARE stage-2a smoke: card (frozen before launch, 2026-10-18)

Question: does Flare start on the Kaggle RTX PRO 6000 from private notebook outputs, parse the dfranzen
harness's tool calls, and emit ≥1.3x the incumbent SGLang's tokens/s under the same closed-loop replay?

## Setup
- Kernel `pranavagrawai/flare-smoke-2a-20261018`, one RTX PRO 6000 session, self-capped at 110 min.
  Expected use is ~60 min of GPU quota (22.8 h left this week).
- Inputs: probe bundle (flare 0.1.0 sha256 7d2ad472…, trixie glibc 2.41 loader), six part outputs
  (`flare-gb202-part{0..4,5b}-20261018`, sha256 per part; whole file LFS c3db5e1a…, checked by the
  CPU verifier kernel), the MTP sweep's replay dataset (`mtp-depth-replay-data-20261006`, 2026-10-01
  control traffic, alpha 0.523).
- Flare: `run PKG --gpus 0 --max-concurrency 16 --mtp-draft-tokens 5 --embedding-table-storage ram`.
  RAM-backed PLE because io_uring is blocked by seccomp (portability probe), so disk PLE can't work.
  Unattended trust: fingerprint sha256(publisher.json key) = 937c0143…; if Flare names a different
  authenticated key, it retries once with that key.
- Functional check: 6 mid-game logged requests at max_tokens 16K, the first alone and the other five
  concurrently. Production payload (temperature 0.7, top-p 0.95, top-k 20, chat_template_kwargs). If
  Flare returns 400, `top_k` and `chat_template_kwargs` are stripped once, recorded, and kept stripped
  for the replay.
- Replay: replay.py unchanged. Checkpoint at run-minute 40, max_tokens = recorded completion,
  S = 10, then 14, then 16 on the same server. Each S has warm-up ≤600 s, a 30 s settle and a 480 s
  measured block. The next S starts only after the previous one's in-flight requests drain.
- Estimators are identical to the SGLang reference: U_resp (completion tokens of responses ending in
  the block / block) and U_overlap (each response's tokens spread over its send→end interval).
  Reference: SGLang M=3, S=10, 2026-10-08 retry1, same harness: U_resp 518.2, U_overlap 477.6.
  A cross-session reference; sub-window CV 0.22 there.

## Decision (on S=10 U_overlap ratio r vs 477.6; U_resp ratio reported beside it)
- **INVALID**: Flare never healthy. Record the cause from the server log: io_uring, trust, memory, the
  CUDA/driver ABI, or load time. A cause we can work around gets one retry; otherwise it's KILL.
- **KILL**: tool calls don't parse (fewer than 3/6 functional rows with a `python` call and nonempty
  `code`, or replay errors >2%), or r < 1.10.
- **PAIRED**: 1.10 ≤ r < 1.30. Before any decision, build a same-session SGLang reference, because
  cross-session noise (CV 0.22, M1 vs M3A spread ~8%) can't separate this band.
- **GO**: r ≥ 1.30 with tool calls parsing. Next is an arm, F1x with Flare serving, for a 60-min no-harm
  public-25 screen vs F1x. Quality (NVFP4/MXFP8-QAD vs W4A16) is only judged there. Swift's INT4
  attention lost 54% of levels.
- S=14/16 are reported for choosing the arm's concurrency, not for the decision.

## Not decided here
License: a private-output copy relies on LIL-1.0 §2.2(b), "storage only You control". The §2.4 email
draft is in PERMISSION-EMAIL-DRAFT.md for Pranav to send or not.
