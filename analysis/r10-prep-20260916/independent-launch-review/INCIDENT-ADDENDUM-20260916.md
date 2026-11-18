# Launch incident addendum — target mismatch; acceptance HOLD

Independent incident classification, 2026-09-16. This supplements the original frozen launch review without changing it. **The receipt reports provider success at a different target. This is an actual launch identity-contract violation, not merely the anticipated advisory-only recovery case. No new approval or identity substitution is granted.**

## Observed evidence

The exact receipt `../r10a-cd82-20260916/push-56e324f3e2004081979fd2f2a9c40e4f.json` records one gate CLI push, return code 0, empty stderr, the title/slug advisory, and:

- Approved/requested target: `pranavagrawai/arc3-r10a-cd82-20260916`.
- Provider-reported success: version **1**, URL `https://www.kaggle.com/code/pranavagrawai/r10a-cd82-constructor-repair`.
- Local command interval: **09:31:38.105667–09:31:41.862893 UTC**, 2026-09-16.

There is one local push receipt in the production run directory. CONSUMED remains present, `used=true`, bound to the expected target and manifest. `launch.json` is absent. Replaying only the frozen gate's read-only `parse_push(receipt.stdout)` reproduces `push URL target/version mismatch`. Parsing fails before the code that writes launch.json and before the later advisory-only check (`r10a-verification/launch_control.py:241-285`). Gate HOLD therefore does **not** mean no remote admission occurred; the receipt reports an admitted version. No SDK recovery result was examined in this addendum.

## Prospective caveat versus realized consequence

Original `REPORT.md:77-81` distinguished two cases: advisory plus an **exact-target** success could retain launch.json for read-only recovery; an unexpected target required holding acceptance. The offline request-construction probe established requested fields only. `test_sdk_advisory_recovery.py:9-13` supplied an expected-target success response, so it did not establish that the provider would preserve the requested slug.

I approved proceeding with that unresolved provider behavior because the gate contained uncertainty and prohibited retries. The incident shows the identity risk materialized. My prospective approval did not establish target preservation and cannot justify retrospective acceptance of this different target. The gate correctly prevents a false expected-target launch record; it cannot prevent a provider-side identity change after admission.

The returned slug equals the title-derived slug identified before launch. That is consistent with title-based slug resolution. The receipt alone does not prove the provider's internal causal rule, whether this created or affected a preexisting object, or the actual version's source/configuration/attachments, status, gameplay, GPU use or cleanup.

## Acceptance and provenance boundary

- **Operational identity acceptance: HOLD.** The returned target is outside the exact frozen target binding. Do not call this a valid launch of the approved identity, rewrite its expected target, create a synthetic expected-target launch.json, or silently transfer approval to the returned slug.
- Read-only recovery of **the reported actual target/version 1** is incident investigation, not an identity amendment. Preserve the chain: frozen manifest/requested target → immutable push receipt → provider-reported actual target/version → independently retrieved metadata/source/status/artifacts. The original report's expected-target lookup is insufficient for this incident; record actual-target recovery separately and never label it expected-target verification.
- Even if readback proves code and settings match, that establishes particular provenance facts; it does not erase the identity deviation or automatically validate this predeclared run. Mechanical/scientific acceptance remains held. **R10b's acceptance gate is not cleared.** This incident is not evidence that the repaired constructor mechanics fail.
- Preserve CONSUMED and the original review. No retry, replacement push, reset, mutation or new launch is authorized by this addendum. Implementer's ongoing read-only recovery can narrow factual uncertainty; no repair/launch proposal is made here.

## Exact bindings

- Push receipt SHA-256: `40ce606c0f820cb8523a739df46b2ddbe5333b8e64126808c33d3e50a8a1ec9c`.
- CONSUMED SHA-256: `ac8c4f7190fdc3aa47109b336f5498c580534fa5ccb59abc8f8601f6315fe542`.
- Manifest: `9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9`.
- Original report SHA-256, unchanged: `9de72876b98627c4a6cab4b356f2d71f8ef4d1ff960a86b597a4364d8e964696`.
- Original launch-review SHA-256, unchanged: `3d842cbfe9c0aa5f8e0986f34e513fe8ce4571cdec528e8df0d6a4f80f678d95`.

Checks: exact receipt/control reads, local receipt count, source trace and parser-only replay; no provider calls. Only this addendum was written.

SUPERSET_WORKER_DONE
 task: R10A-LAUNCH-INCIDENT-CLASSIFICATION
 summary: Actual returned-target mismatch exceeds advisory-only case; expected-identity and mechanical/scientific acceptance HOLD. Prior approval preserved, not transferred.
 files: analysis/r10-prep-20260916/independent-launch-review/INCIDENT-ADDENDUM-20260916.md
 checks: Receipt/control hashes; original review integrity; parser-only mismatch reproduction; no remote calls or mutations.
 handoff: Preserve consumed authority and exact provenance; await separate read-only actual-version recovery evidence. No new launch approval; R10b gate remains uncleared.
