# 0158 LS20 live run — inline result

## Result

The single approved, non-resumable LS20 run completed with **0 actions** and **0 levels completed**. It did not test grounding or gameplay success.

- Run: `artifacts/live-ls20/20260812T123422Z-gemma31b-inline-result`
- Run ID: `320badf242b04d928cb40676d5eb0303`
- Card SHA-256: `701bb9bc6a690ec757d9944fabac288364fcc86cc00a344d2b5e8e4474c93676`
- Provider route: `coreweave/bf16`, fallback disabled
- Scorecard: terminal `PAUSED_FOR_REVIEW`, non-resumable

## Observation

The one provider crossing returned HTTP 200, not 429. The provider marked the response `incomplete` with `reason=max_output_tokens`. Reported usage was 662 input tokens plus the full 8,192 output-token allowance (8,854 counted total; 2,127 reasoning tokens reported separately).

Gemma emitted reasoning only and made zero tool calls. It recognized that `frames()` was available, but then began spelling the large grid out row-by-row in its reasoning. It exhausted the generation limit before calling the sole provider-visible tool, `execute_python`, so neither `make_mask()` nor inline `return_result(candidate)` was exercised.

Therefore this run distinguishes the current failure from the prior result-tool convention failure: the direct `return_result` tool was successfully absent, but the model never entered CodeAct at all.

## Classification correction

The immutable failure packet called this `substrate_failure`. That generic label is contradicted by the audited HTTP 200 response and provider completion metadata. A separate immutable `classification-correction.json` records the corrected class: **`model_generation / model_output_exhaustion`**. The original packet was not edited, and its checksum manifest still verifies.

## Between-run diagnostic fix

After snapshotting all 20 executed candidate files, the host was changed so a worker `GenerationError` is classified as `model_generation` rather than `substrate`. Failure-packet callable-surface records now include the newly exposed `make_mask` helper. These changes were made after the run and did not affect its result.

## Next discriminating intervention

Do **not** merely increase the token limit again: the model used the extra capacity to serialize more of the frame. The next candidate should change the first-step convention so Gemma inspects or computes over the frame inside `execute_python` without copying it into reasoning—for example, a compact component/statistics inspection helper or an explicit first-call instruction. That is a new candidate and requires a new card and one-run approval.

## Verification

- Candidate source snapshot: 20 files, each checked against the launch manifest before copying.
- Failure packet: every entry passes `sha256sum -c SHA256SUMS`.
- Post-run tests: 9 host/adapter tests passed; 9 NOOA runtime tests passed.
- Ruff: host, runner, tools, tests, and runtime passed.
- Provenance audit: passed; 7 sources and 4 patches verified; no advanced capability modules.
- Approval consumption exists; the trajectory cannot be resumed.
