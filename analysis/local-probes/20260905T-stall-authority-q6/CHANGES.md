# q6 contract repair — 2026-09-05

Status: implementation and local contract qualification complete; Kaggle preflight and experiment unlaunched.

The Kaggle host now supplies the evidence that the inherited q3c gates require. Queue tails stop at level advancement, no impact, and terminal outcomes. Completed-level frames reach the scribe before the next state is announced. Observation history, prediction results and raw transitions are recorded in native order. Worker death ends its actor subprocess; pending actions are not replayed through an implicit restart.

`patched-source/after/host_contract.py` owns host evidence. `stall_contract.py` owns the detector used by the actor, historical replay adapter and independent scorer. The detector compares the current rationale with six preceding committed batches, after 25 actions, at similarity 0.55. Control records eligibility; treatment requires an explicit resolution.

A resolution commits exactly one legal action. Continue requires a falsifier, probe requires distinct predicted outcomes, and switch requires a changed basis. Full bases survive logging. Eight rejected resolutions latch an abort. The authority record is committed before the action queue wakes the host. The host records the actual consequence even when terminal state or budget prevents another actor invocation; missing authoring is censored, not credited as a revision.

The note mutation was caused by the redaction prefix list containing `/` for shallow run paths. Excluding the filesystem root fixes that transformation. The new integrity guard checks stored note content, captured readback and the actual rendered feedback passed to the next model request. Detected mutation latches an abort. Complete printed readbacks are covered; partial, unused or evicted reads are reported as unobserved, and arbitrary direct filesystem reads are outside this guard's coverage.

`score_q6.py` recomputes detector events, outcomes, falsifiers, recurrence, authority changes and P1–P4 holds from raw artifacts. It rejects missing/contradictory evidence and scorer/source drift. P9/P10 require the full 20-action follow-up horizon unless recurrence already establishes failure. P11 is enforced. No continue dispositions means P10/P12 are not applicable, rather than an automatic failure. Episode denominators are retained; stable run and unique session identities prevent copies or restarts from becoming extra replicates.

`deployment/kaggle/q6-nooa-candidate` is a new candidate; the historical s0 candidate and q6 before snapshot were preserved. Its manifest pins runtime files, adapter, scorer, card and builders. Both deterministic notebooks contain every file checked at worker startup and differ only in the treatment flag. Shared runtime changes for q6 only forward the treatment flag and run identity.

Evidence and exact commands: [implementation/RESULT.md](implementation/RESULT.md). The saved scripted traces are qualification fixtures, not experimental results.
