# Independent exact-byte package review — PASS

Task: OR-CONSTRUCTOR-REASONING-PAIR. Scope: candidate package only. No provider API, credential loader, paid request, production entry, or release was invoked. All reviewer writes are confined to independent-review/. Existing R10 HOLD remains unchanged.

## Exact binding and disposition

**PASS** for inventory SHA-256 `fa134897d4c592335ef9f7f047838773bf0c5cb4e74ca24921ee6d402c215dee`, endpoint `reka/fp8`. This is the independent package-review gate only. Explicit PM credential-use approval and Hermes RELEASE are still required before execution. It is neither a hosted-treatment result nor a guarantee of remote enforcement.

The inventoried files were independently rehashed before and after checks. All match. The 15 source bindings match their R10 donors; only constructor.py, constructor_contract.py and request_contract.py differ in the copied engine. Changes agree with engine.diff. Prefix bytes select exactly the original B step-1 RIGHT transition; original first three messages equal raw constructor request 074. The user evidence packet matches. Rulings match /tmp/or-constructor-pair-rulings.md byte-for-byte. BINDINGS.json records the complete inventory and these checks. No live/ directory exists at review completion.

Key bindings:

- Card: `fd252b34c0b0cba3fac4a355926acf63745d47aa56251c09b01f0d2b514f2e51`
- run_pair.py: `3423ba0ae543bb34fd449274d67526c318b893afb5086888bc02ec4054aac964`
- transport.py: `7bebcfb1ca9725b8eec76d995eae94764dc4d53f668394e30662b3a4aba76f3e`
- protocol.py: `7e6c5f8dbbedcecfdd2f79e746c255f3fc2e0fb6a45c00737da9cb09c70ba9ca`
- test_pair.py: `8ae27d24de9ecdd7b531afe3210eefb6eae45b245729fa4cd17887e35eaac2a6`

Changed candidate/inventory bytes invalidate this PASS. Discovery is retained candidate evidence; it was not refreshed through an API in this review, per instruction.

## Findings

No blocking defect found within this bounded package review.

1. **PM rulings are implemented.** Shared prompt removes the stale reasoning-fatal clause and truthfully distinguishes hosted accounting. Shared runtime-envelope changes remove obsolete Kaggle/relay disclosures. Source-format wording, schema, evaluator and evidence remain unchanged. The first telemetry value is fixed to 240 for exact start parity; actual deadlines still use monotonic time. On/off subsequent trajectories may diverge. This is a same-adapted-prompt comparison, not byte-identical historical replay. See engine.diff; transport.py:59–64 checks actual first outbound-body parity before production on dispatch.
2. **Routing and treatment remain explicit.** transport.py:9–15 pins the singleton full Reka FP8 tag, price filters, required parameters and no fallback; wire() changes only declared transport fields. No extra parallel_tool_calls, SGLang flags, sampling knobs or repair plugins appear. Required tool-choice support is present in retained selected endpoint metadata. Raw outbound bytes are authoritative, not the logical engine's off marker. Returned model/provider/direct routing/attempt/pipeline evidence is checked before tools. Explicit zero reasoning is required in off; on must have nonzero reported reasoning with payload somewhere in its episode. Missing counts or contradictions fail closed. A successful HTTP response alone is insufficient.
3. **Continuation is preserved through the real loop.** protocol.py retains ordered reasoning_details with opaque fields, or a reasoning string when details are absent, and retains null content plus tool IDs/arguments. It removes response-only tool indices rather than mutating function arguments. An unsupported reasoning_content-only response stops instead of silently translating. Fixtures run details-only and plaintext reasoning through inspection, failed stringified-list submission, exact NameError feedback, next request, repaired source and sealed ABI freeze. This establishes local protocol mechanics, not that Reka will honor the flags.
4. **Budget logic uses the PM-approved exposure interpretation.** transport.py:49–58 reserves request bytes plus framing at pinned input rate and 131072 advertised completion tokens at pinned output rate (completion exposure $0.3342336), not merely requested 4096. It writes possibly-dispatched attempt and unknown exposure before HTTP. Validated actual cost/usage releases unused reservation; missing/invalid usage leaves exposure unresolved. Tokens are counted once, including reasoning; observed >4096, cost/exposure overrun or truncation stops the pair. Shared $2 admission, separate 6-request/600000-prompt/24576-completion limits, endpoint context and deadline checks are present. No cache saving is assumed. Neither reservation nor local cutoff guarantees remote compliance or cancellation; the card states that limit.
5. **Local containment/publication is bounded.** run_pair.py:66–80 owns/reaps the worker group. Main's lines 111–123 inspect timeout, exit, stopped ledger and unknown exposure before root promotion or proceeding to on. Child-only freezes remain forensic on failure. The engine/evaluation/inspection copies retain the sealed boundary; fresh mapped-user fixtures exercise host-file/network/fork/write denial, ordinary candidate failure/repair, inspection/submission ceilings and in-flight unknown exposure.
6. **Authority remains separate.** run_pair.py:84–86 checks independent package and Hermes artifacts, then PM credential approval before loader use. Exclusive live/ and per-arm consumption markers prevent ordinary re-entry. The prior implementer credential-file availability read is disclosed in AUTH-ACCESS-NOTE.md; this review does not retroactively authorize it or independently certify its historical narrative. The reviewer did not call the loader or inspect credentials. Credential permission is still pending.

## Independent validation

All commands below run from repository root. Logs and temporary fixture artifacts are under this review directory.

```bash
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-reasoning-20260916T185550Z/test_pair.py --basetemp=analysis/openrouter-constructor-reasoning-20260916T185550Z/independent-review/basetemp-1
PYTHONDONTWRITEBYTECODE=1 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/openrouter-constructor-reasoning-20260916T185550Z/independent-review/test_parent.py --basetemp=analysis/openrouter-constructor-reasoning-20260916T185550Z/independent-review/basetemp-parent
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/openrouter-constructor-reasoning-20260916T185550Z/check_candidate.py
```

- Candidate: **34 passed in 19.88s**, tests.log.
- Independent parent checks: **3 passed in 0.31s**, parent-tests.log.
- Local candidate verifier: matching inventory, 15 unchanged donor sources, exact prefix packet, no consumed live state.

Additional tests exercise the actual main orchestration with HERE redirected into reviewer temporary directories, gate checking isolated, and auth replaced by an inert fixture. Real child subprocesses write forensic artifacts and time out or exit with unresolved exposure. Main blocks root promotion and never starts on. A separate test proves credential denial happens before any loader invocation. These are deliberately local tests, not a production launch or remote cancellation test. No real gate artifact was fabricated for Hermes or PM. Use fresh basetemps for any later repetition.

## Limits and next permitted step

Hosted reasoning toggle, returned usage/routing shape, exact provider behavior and actual cost are untested. Strict response validation may stop on the first authorized arm request; that is an intended bounded outcome, not permission to retry or relax checks. Serving identity remains provider-reported hosted FP8, not a hash-equivalent Kaggle checkpoint. A single ordered pair supports descriptive constructor outcomes only; an accepted freeze is unscored visible ABI replay, not game-model accuracy or efficacy.

The dispatch/rulings settle remote-cap uncertainty; this review does not demand an impossible pre-spend provider guarantee. The candidate must still stop and retain evidence if observed behavior violates the declared threshold or reservations.

Next permitted decision is Hermes review/release and explicit PM credential-use approval for these exact bytes. This PASS alone authorizes neither credential access nor paid execution. Do not change endpoint/prompt/source/budget after start, reset consumption, or clear R10/R10b gates.
