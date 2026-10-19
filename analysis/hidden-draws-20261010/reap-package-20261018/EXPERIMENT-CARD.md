# REAP448 + c18 competition-packaging qualification

Question: Does the minimally repaired REAP448 + c18 notebook finish a bounded
public qualification while retaining the tested loader, mask, drafter, context,
sampling and serving settings?

Posture: Operational qualification of a packaging successor, not a new score test.

Authority: Pranav authorized one REAP hidden draw on 2026-10-18 and explicitly
approved this additional qualification: "Run the bounded qualification (Recommended)".
The approval covers one launch with the 2,700-second cap and conditional queueing.

Candidate: `pranavagrawai/reap448-c18-hidden-ready-20261018`, intended version 1.
Notebook SHA256: `b0e3617055aee24555fa94a33ec8e257f19ec31353c2579d135b149f6c39e0e1`.
Parent: `pranavagrawai/prune-retain-candidate2-20261011` v1, kernel 138353174.
Parent notebook SHA256: `16d3956e154108d5d49407dd19fe43d442974cde5585ec6b764308fd3fbb78d7`.

Changes: Remove the public-only competition assertion; remove the unconditional
7,254-second overwrite after the competition runtime branch; stop interactive play
after 900 seconds from benchmark start. The latter is inactive in competition mode.
All other code cells, including the complete serving/pruning cell and embedded mask,
are byte-identical. `packaging.patch` is the complete executable-source diff.

Contrast: None. The parent already has a completed full public-25 comparison
(106 vs 103 levels; inconclusive). This run cannot estimate a new treatment effect.

Prediction: The retained REAP loader and native 512-expert drafter start and serve
18 streams; the unchanged solver plays fresh public games and exits cleanly at the
interactive soft deadline. No directional level or score prediction is registered.

Measure: Remote identity and all code cells; terminal kernel status; raw kernel log;
`pruning/target.json`, `pruning/draft.json`, server configuration; `benchmark.json`
with exactly 25 game identities, no crashed games and at least one real native action;
`submission.parquet`; and explicit confirmation that the qualification reached
benchmark execution. Independently reconcile completed levels and action histories.

Cap: Exactly one launch, one RTX PRO 6000, at most 2,700 GPU-seconds including startup,
900 seconds of interactive play. Zero paid compute, zero training and zero competition
submissions under this card. Require sufficient live quota and a free GPU slot before
launch. Do not resolve contradictory quota data by assuming permission or availability.

Stop: Platform hard cap or normal notebook completion. Startup failure and missing
qualification artifacts stop the path. Preserve output and spend; no automatic retry.

Invalid if: Source/metadata drift; missing or wrong target/draft identities; changed
sampling/serving settings; crashed or missing public game; absent benchmark execution;
missing real action evidence; or timeout before normal completion. A packaging
failure says nothing about REAP's comparative gameplay quality.

Result map: Valid COMPLETE plus all gates -> eligible to queue exactly that version
for the separately authorized one hidden draw. Failure/invalidity -> no queue entry,
preserve evidence and diagnose without automatic relaunch. A low public level count
is not a rejection gate for this 15-minute operational check.

Outputs: This directory's source, metadata, patch, VERIFICATION.json, launch receipt,
collection manifest, raw qualification outputs and subsequent exact-version readback.

Limits: Public execution cannot exercise the actual competition gateway. CPU execution
of the real config/deadline branches verifies the two repairs; the hidden draw will
still be the first live competition-mode use of this successor.
