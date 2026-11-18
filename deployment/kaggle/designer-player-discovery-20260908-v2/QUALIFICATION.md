# Reasoning-enabled candidate: evidence interface successor

Current interface contract:
`6db629026e9681f27cd91a56463a8f16bf2ef1dbf92d49b1d852a9ec8d07f3a6`.
Adds measured last-transition access and execution-only runtime-parity helper
preflight, with corrected Actor-facing documentation. See
`analysis/deployment-probes/20260907-designer-player-interface-repair/RESULT.md`.
This successor has not had a paid run or Kaggle launch. The history below belongs
to earlier contracts and is not live qualification of these interface changes.

## Previous lifecycle qualification

The failed live version is preserved in the original result directory, including
a source archive verified against its `f26e4a957111...` manifest. The lifecycle
contract was `6b8a5c1d6955c09d2f324f9ce806539fceec37254e9b7832de8f604dec89733e`.
The dispatcher and deadline/cleanup repairs are locally verified, including
real subprocess failures from the notebook-extracted payload. See
`analysis/deployment-probes/20260907-designer-player-lifecycle-repair/RESULT.md`.
The authorized live recheck exercised the deadline boundary successfully: no
post-close admission, 18 responses accounted for, one completed after shutdown
and explicitly not delivered. Overall action qualification remained incomplete
(one of two required actions; mindset not started). See
`analysis/deployment-probes/20260907-designer-player-reasoning-recheck/RESULT.md`.
No Kaggle launch has been performed. The generation-error fail-fast path remains
locally tested but was not naturally exercised in that live recheck.

Reasoning is explicitly enabled in both arms. This is a separate successor to
the thinking-off candidate; the previous run and its evidence are unchanged.

## Verified locally

- Repaired candidate: 53 tests passed (one existing deprecation warning), including
  dispatcher, admission, interruption, shutdown and extracted-failure tests.
- Real NOOA client parsing accepts both reasoning response fields and a valid
  tool call with null prose, without issuing an empty-content retry.
- Actual subprocess worker, adapter, and loopback model exercised a reasoning
  response followed by an attributable action and recorder entry.
- Explicitly disabled, missing, and conflicting reasoning configurations fail.
- The effective total output allowance is 16,384 tokens, including reasoning.
- Packaged execution tests pass. `prepare_candidate.py --check` passes.
- Mocked Kaggle CLI verification passes with zero network calls, competition
  `arc-prize-2026-arc-agi-3`, and RTX PRO 6000 selection.

Kaggle Actor HTTP and pending-worker limits are 900 seconds. The separately
bounded OpenRouter pilot uses 300-second HTTP and 900-second arm limits. Neither
the compatibility warmup nor explicit flags alone establish generated reasoning.

## Live evidence and release status

See `analysis/deployment-probes/20260907-designer-player-reasoning-openrouter/`
for the frozen card, request/response records, worker evidence, and final result.
Live reasoning and tool-call compatibility have been observed. The loop arm
reached the output limit without a tool call and exposed a separate dispatcher
failure: after logging a retry, it waits for an already-consumed notification.
It was stopped explicitly after the request finished, with no request in flight.

The mindset arm reached one real DOWN action, but its per-action check was
unresolved (`current prediction is not committed`). Its 900-second outer alarm
then entered cleanup without retiring the worker, which admitted another request.
An explicit ABORT stopped local work. The relay retained the final attempt's
outcome separately; see the live result for terminal and delivery accounting.
This timeout/cleanup failure invalidates the planned bounded comparison.

This candidate is **not launch-qualified**. The notebook is built but has not
been uploaded or launched. Do not interpret reasoning compatibility as evidence
of improved gameplay or of target-GPU performance. The live recheck demonstrated
deadline closure, but did not meet the two-action gate; integrated promotion
remains incomplete. No failed live result has been relabelled as a pass.
