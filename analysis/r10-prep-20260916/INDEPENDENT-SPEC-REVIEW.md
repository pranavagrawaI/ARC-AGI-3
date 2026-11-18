# R10-SPEC-REVIEW — independent specification findings

Reviewer posture: Karl Popper — require a possible observation that distinguishes each asserted outcome. Date: 2026-09-16. **Specification review completed; blocking contracts below remain unresolved. This is not package approval, freeze approval, or launch approval.**

## Scope and authority

Read the dispatch, handoff, and full R10b draft; traced only relevant R9-cd82 constructor, action overlay, limits, and experiment-scoring boundaries. Repo access was read-only. Only this report was written. Consulted the shared board read-only; no messages, tracker changes, tests, launches, git mutations, or remote calls. Used the code-review skill. Memory registry search yielded no R8/R9/R10 hits; findings use current files, not memory. No exhaustive predecessor audit or review of newly implemented R10 package bytes was performed.

Latest user instruction and `/tmp/r10-approved-worker-dispatch.md:3` supersede the stale pending-approval statements at `/tmp/r10-orchestration-handoff-20260916.md:42-56` and `analysis/r10-consolidation-draft/R10B-EXPERIMENT-CARD-draft.md:70`: extra compute and conditional private pushes are already approved. PM game selection is CD82 for R10a, s5i5 local regression, LS20 for R10b. Do not ask for those approvals again. R10b preparation is permitted now; its run remains gated on Hermes verifying R10a mechanical acceptance. Draft line 3's restriction on creating an identity card should be reconciled with dispatch lines 3-4 permitting preparation now; this is a documentation fix, not a new user decision.

Citation shorthand below: **D** = `analysis/r10-consolidation-draft/R10B-EXPERIMENT-CARD-draft.md`; **P** = `deployment/kaggle/actor-critic-construction-kaggle-r9-cd82`. Each shorthand plus filename/line denotes that exact repository path.

## Findings requiring an intent ruling only where the proposed interpretation is not accepted

### 1. HIGH — Literal cross-arm information parity conflicts with independent gameplay

**Evidence:** D:5,9-12,34-35. Arm B learns from its own actions, while line 34 literally requires its packet to be a strict subset of what arm A can retrieve. Once actions differ, B's executed transitions need not exist in A's history. Including B's current theory also cannot satisfy literal raw-input parity: that theory is the declared treatment (D:15,20,47).

**Consequence:** Ordinary treatment-induced divergence can be labeled invalid, or an implementer can quietly leak B evidence to A to satisfy the literal rule. Packet hashes alone (D:48) establish identity, not availability or information provenance.

**Minimal resolution:** Define parity as equal evidence-access capabilities over each arm's own executed prefix, with the same fields and retrieval APIs; B's packet contains only its own Actor-retrievable raw evidence plus its previously accepted theory. Explicitly declare prior theory and its Actor context block as treatment-derived content. Retain full packets, source file/record hashes, arm/session/step identities, cutoff, retrieval provenance, and request receipts; exclude future records, game source and cross-arm evidence. Use an allowlist/provenance audit, not hash equality between divergent trajectories.

**Owner:** This operational reading changes literal line 34 and must be accepted by the intent owner (PM if delegated, otherwise user). Capability parity is the recommended reading. Replay/shared-evidence designs would change the experiment and need a separate ruling; do not silently build one.

### 2. HIGH — A failed typed prediction does not mechanically contradict a prose claim

**Evidence:** D:25,30; P/nooa-join/chassis/solver_agent.py:1594-1603 explicitly limits host scoring to typed predictions. P/nooa-join/chassis/decision_packet.py:100-139 emits `measured_prediction_only`, retains conjectural updates, and does not connect a result to WORLD_THEORY claims.

**Consequence:** There is no defined host rule for deciding which supported claim a mismatched prediction contradicts. Matching strings or declaring every failed experiment a theory contradiction silently invents semantic authority.

**Minimal resolution:** Preserve all four triggers. Record a prospective link from a supported claim ID/version to an experiment ID, specific typed prediction and applicability scope before execution. The host can detect that this explicitly linked prediction failed on its matching transition; log this as counterevidence to a model-declared implication and request consolidation. It must not automatically certify/retract the broad prose claim. No valid link means this trigger is inapplicable, with Actor/periodic triggers still available. Link metadata can live in a sidecar record so the existing set_experiment contract stays intact. Do not require every theory claim to be executable.

**Owner:** Mechanical once this limited meaning of “host-detected contradiction” is accepted. If the intended requirement is semantic contradiction detection for arbitrary prose, that is a user-decision blocker: no such oracle is authorized. No new semantic evaluator should be invented to close it.

### 3. HIGH — “Trustworthy scoring” needs an explicit scope for R10a

**Evidence:** `/tmp/r10-approved-worker-dispatch.md:4`; P/constructor_contract.py:14-22,35-39 states that freeze is ABI replay only, commitments are not compared with observations, and assimilation receives exactly `{"status":"unscored"}`. D:10 retains independent set_experiment scoring.

**Consequence:** Counting a successful query/assimilation as a correct prediction would falsely pass R10a. Conversely, requiring semantic scoring of arbitrary frozen commitments would expand the approved repair into a new evaluator.

**Minimal resolution:** R10a acceptance must distinguish (a) truthful lifecycle/digest/transition binding, (b) reproducible existing typed set_experiment scores, (c) frozen commitment accuracy, which remains unscored unless a separately specified observable contract exists. Preserve raw prediction, pre-state, exact action including coordinates, executed successor and digests; report `unscored` without converting it into match, zero error, or successful dynamics. Reconcile gameplay progress to official recordings independently.

**Owner:** Recommended mechanical interpretation follows the existing-treatment claim ceiling. Only an insistence that R10a add semantic frozen-model accuracy scoring requires a further intent ruling and bounded scoring specification. Do not presume that expansion is required.

## Mechanical specification blockers — resolve in the card/package without reopening user approval

### 4. HIGH — Claim identity and revision schema cannot support the required trace as written

**Evidence:** D:19-25 requires claims carry “exactly” six fields, omitting claim ID; D:47,50-51 requires full claim lists, survival tracking and claim-ID use detection. Top-level semantic buckets do not specify whether their free text is subject to claim validation.

**Minimal resolution:** Put stable host-managed IDs in an envelope/map around the exact six-field claim payload, or explicitly amend the field list. Define typed containers and use claim references for every substantive assertion in entities/mechanics/goals/persistent_state and the status buckets; reject uncited assertions hidden there. Define version genesis, single parent, scoped evidence references, timestamp/step meaning, and identity retention across rewording/retraction. A changed meaning needs explicit replaces/splits/merges lineage, not phrase matching. Validate references exist in the supplied arm/session prefix; a nonempty fabricated reference is insufficient. Unknowns must cite evidence revealing the gap, as line 25 requires. Structural validation does not certify that citations entail the prose. Rejected revisions leave the prior accepted version active and are logged.

### 5. HIGH — The +120 s allowance and 12 × 40 s calls lack a common wall-accounting rule

**Evidence:** D:39-43,60 allows up to 480 s of consolidator call wall, extends admission by at most 120 s, promises symmetric Actor time, and invalidates wall/cap breaches. P/gameplay_play.py:42-53,456-466 uses a concrete admission deadline and separate request reserves; simply increasing one timeout is not sufficient.

**Minimal resolution:** State whether the 120 s is only compensation or also a cumulative blocking budget. Recommended: serial consolidation with at most 120 s of total compensated blocking time; stop admission before that purse is exhausted, limit each hard deadline by its remaining purse, and preserve 600 s Actor admission eligibility. Define measured interval including queue wait, packet preparation/validation and cancellation cleanup; credit only time actually blocking the Actor, never overlapping intervals twice. Preserve raw elapsed times and actual deadline updates. Define 30 s soft timeout behavior and 40 s hard cancellation/late-result rejection; failed/rejected/timed-out attempts consume the call purse. Count Actor, critic and summarizer calls within the 129 shared incumbent purse, consistent with P/gameplay_play.py:47-49,520; consolidator alone gets the extra 12. Warmup + both incumbent purses + consolidator <=271. Define attempted versus server-admitted counters and reconcile receipts. No uncounted repair/retry calls. If compensation-only accounting is selected, explicitly disclose possible Actor time loss above 120 s; do not claim equal available Actor time.

### 6. HIGH — Larger constructor limits do not prove inspection and ordinary repair are reachable

**Evidence:** Dispatch:4 requires actual inspection and repair through real entry points. P/constructor.py:197-208 can remove inspection based on request, token and wall reserves; :304-348 parses and evaluates source; :349-363 returns failure feedback or immediately accepts. P/construction_service.py:42-47 and P/gameplay_play.py:50-52 retain separate 45 s/two-request and hard-kill limits.

**Minimal resolution:** Specify the acceptance trace: real production entry point → offered inspection tool → sealed prefix inspection → observed output returned in the next model request → malformed candidate through actual JSON/source transport → rejection feedback → one corrected ordinary submission → accepted immutable freeze → query → exact executed successor → assimilation. Use deterministic scripted transport for a bounded local regression, retaining real subprocess/serialization/evaluator/IPC paths. Include syntax failure, literal backslash-n transport failure, and a legitimate string literal containing backslash-n so a global replacement cannot masquerade as robust serialization. Pin enough requests, tokens, submissions, wall and outer hard timeout for that path; verify effective limits at runtime. A helper unit test or raising only the card numbers is insufficient. CD82 must cover legal CLICK coordinates and non-CLICK actions; s5i5 is local regression.

Clarify the R10a gate evidence split: deterministic local traces establish repair capability without requiring a naturally occurring live syntax failure; collected CD82 evidence must establish actual freeze, prospective queries, transition binding and retained host events. If live repair occurrence is mandatory, that different acceptance requirement needs to be stated before the run. Hermes must record local capability and observed live behavior separately, never substitute fixture success for remote execution.

### 7. MEDIUM — Post-selection query telemetry cannot establish same-action decision leverage

**Evidence:** Dispatch:4; P/live_agent.py:55-90 queries only after receiving the selected action and immediately calls super.submit_actions, with no intervening Actor decision. P/live_agent.py:15-22 also exposes explicit Actor-invokable queries, a distinct path.

**Minimal resolution:** Log selection, query, Actor exposure and execution order, plus whether a further Actor decision opportunity existed. Automatic same-action query leverage is `not available on this path`, not a measured failed consultation. Measure explicit preselection consultation separately if it actually occurs, and later references/changed proposals as observable use proxies. Report denominators and missing opportunities; never infer causal benefit from a referenced phrase or causal inertness from zero opportunities. Add no veto, extra confirmation or new consultation architecture to repair this metric.

### 8. MEDIUM — Trigger and Tier 1 measurement ambiguity can change the reported outcome

**Evidence:** D:27-31,48-52,60,63-65 leaves trigger collisions, reset of the 12-transition clock, exhausted-purse events, survival horizon N, region identity and “independent use” unspecified. It calls string matching Actor consumption and makes reduced duplicates/latency sufficient for an adds-value conclusion.

**Minimal resolution:** Coalesce simultaneous trigger reasons into one call; log every reason, accepted/rejected/skipped outcome and purse state. Define which attempted/accepted invocation resets the periodic counter and prevent consolidate() loops without new evidence. Define Actor-context snapshot freshness at request assembly; record the exact version rendered and serialize publication so normal in-flight requests are not mislabeled stale.

Predeclare survival horizons, duplicate key/region and eligible denominator, later-use rubric, and right-censoring for claims never reused before termination. Repeating a no-change probe does not itself prove no new information; label it an operational proxy. Claim-ID/text matches establish references, not causal influence; distinguish mechanically detected references from reviewed evidence of use and record the same discovery/use opportunity measures in control A without WORLD_THEORY. Unsupported “mechanically applicable” prose remains unclassified. Limit one sequential LS20 pair's result to descriptive evidence consistent with added-compute consolidation benefit; it does not isolate consolidation from extra compute or establish replicated efficacy. Preserve the required “cross-level hypothesis not tested” when no qualifying boundary occurs.

## Handoff and verification

Resolve findings 1-3 as explicit contract interpretations; ask the user only if intent remains materially different from the minimal readings above. Findings 4-8 are implementation/card choices with concrete acceptance evidence, not requests for new compute or launch authority. Implementer can continue independent preparation while these are reconciled.

Next reviews remain separate: independent package-byte review → manifest freeze through existing machinery → exact frozen launch review → Hermes gate/hash checks → at most the conditionally authorized private push. R10b run additionally requires Hermes' evidence-backed R10a mechanical acceptance. This report supplies none of those approvals. Reviewer remains available for later exact-byte review.

Checks performed: full reads of the three requested documents; targeted line-numbered source tracing; read-only board listing; source-document SHA-256 snapshot below. No runtime qualification or containment tests executed, and none claimed to pass. Initial guessed source paths were corrected before making findings.

Reviewed document hashes:
- `/tmp/r10-approved-worker-dispatch.md`: `a059750647a2fad59f0cd03d5348e0d9db01761db8d9c63c557398ae34c9cec8`
- `/tmp/r10-orchestration-handoff-20260916.md`: `8a8746a0eedea94e80689a5a167d369bbed94497068d69b6bd2f1661d3f2063b`
- D: `847f864949ecc79894687df3ae0bd8c6e687ce24b26c7fbc497c28f121283ddc`

SUPERSET_WORKER_DONE
 task: R10-SPEC-REVIEW
 summary: Independent spec review complete; eight bounded findings, with intent interpretations separated from mechanical choices. No package or launch approval.
 files: /tmp/r10-independent-spec-review.md
 checks: Read-only document/source review and source-document hashes; no tests or remote qualification.
 handoff: Resolve cited contracts in reviewable R10 artifacts; preserve all package/freeze/launch/Hermes and R10a-before-R10b gates. Available for exact-byte review.
