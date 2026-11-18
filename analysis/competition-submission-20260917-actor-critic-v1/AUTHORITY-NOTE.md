# Authority note — T5 sole launch owner (2026-09-17)

**Human authority.** The user requested: "I want to make a competition submission … with the actor critic solver." Task T5 (`/tmp/arc_submission_t5.txt`) construes this as authorization for exactly one Kaggle kernel push and exactly one ARC-AGI-3 competition submission of the independently approved exact candidate, within the frozen card caps. It does not authorize retries, a second version, changed bytes, widened budgets, a different target, or another daily submission.

**What `authorization.json` binds.** Identity `arc3-ac-competition-20260917-v1`, target `pranavagrawai/arc3-ac-competition-20260917-v1`, manifest `dcb716bc…`, card `eb8ea216…`, notebook `3f38ad9b…`, the exact absolute run dir and push command, `max_pushes=1`, `max_submissions=1`, `timeout_seconds=32400`, `used=false`. It carries only the fields `launch_control.py` validates; this note carries the rest.

**Independent review bound.** `final-review.json` SHA-256 `b18ee233d241cdf254aa894cb8686c760ba49e73c2f58b86d0e1afa6343ed0be`, status APPROVED, independent, review task T4R. Its residuals are accepted as known limitations, not resolved.

**Waiver: no exact-byte remote public-mode run.** This package has no public-mode path (the notebook's non-rerun branch writes a dummy parquet only; the rerun branch requires the competition gateway). The absence of a remote public-mode qualification of these exact bytes is accepted for this launch. It is recorded here as a limitation of the evidence, not as evidence.

**Preflight.** `LIVE-PREFLIGHT.json` (2026-09-17T11:16:02Z) reports GO: hashes exact, gate reconciliation OK over 85 sources, `numAllowedNow=1`, no prior submissions, GPU quota 25.92 h of 30 h, authenticated as owner, target absent (403 on own slug, calibrated against an existing private kernel that resolves; slug search empty), metadata exact, CLI 2.2.4 with `--timeout`/`--accelerator`.

**Submission authorization.** `SUBMISSION-AUTHORIZATION.json` is separate. It becomes valid only for the exact owner/slug/version returned by the sole push, after server readback passes and the commit completes with `submission.parquet`. One submission; no retry on ambiguous admission.

**Known remote risk, not a preflight blocker.** On 2026-09-03 (commit 19851bf) Kaggle recorded `machine_shape=NvidiaRtxPro6000` but allocated a P100; later R8/R9 runs (2026-09-15) did receive the requested shape. The commit run of this notebook does not need the GPU; the competition rerun does. If the rerun lands on the wrong shape, that is a remote outcome to preserve and diagnose, not a reason to retry.
