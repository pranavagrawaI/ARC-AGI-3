# R10b — consolidation A/B — UNFROZEN preparation draft

Status: UNFROZEN preparation card, 2026-09-16. User approved LS20, additional compute, capability parity, and linked typed counterevidence. R10a's outcome gates RUN, not preparation. No freeze or launch until subsequently instructed; independent package/launch review and Hermes verification remain mandatory.

Question: On public LS20 under the pinned official `Agent.main` loop, with the same Actor, same model, and symmetric per-arm Actor budgets, does an additively maintained, versioned WORLD_THEORY — compiled by a dedicated consolidator call from evidence the Actor can already access — improve within-level consolidation outcomes (duplicate-probe rate, rediscovery latency, contradiction persistence, retrieval and use of consolidated claims) relative to the identical Actor with the existing evidence machinery alone?

Posture: this is a NEW experiment, not "constructor v2". No frozen predictor, no sealed RPC, no prospective branching, no automatic action intervention, and no requirement that any part of the theory be executable. The treatment is exactly one new capability: an immutable-evidence → consolidator → versioned WORLD_THEORY → normal-Actor-context loop. The historical context: R8/R9 measured the construction join (valid three times; no measurable effect on actions; the constructor is upstream of nothing and its admission test is well-formedness only). This card tests the knowledge-integration hypothesis the constructor was originally meant to serve.

Contrast: two arms in ONE kernel, same served model, run sequentially after a shared warmup.
- A (control): the unmodified chassis — Actor + existing evidence machinery (states.jsonl, transitions.jsonl, decision_packet, set_experiment scoring, objection ledger, epistemic critic, save_procedure, render_measured_evidence) exactly as in R9.
- B (treatment): identical, plus: (1) a consolidator model call — same served model, separate request stream, its own system prompt (below); (2) a versioned WORLD_THEORY object persisted in the run dir; (3) the current theory rendered into the Actor's context through the existing context-block mechanism; (4) an Actor-invokable `consolidate()` method.
The arms differ only in the additive consolidation capability and its predeclared extra compute. set_experiment, objection ledger, critic, discovery arms, and every other chassis capability stay ON in both arms; the treatment is purely additive. Removing or weakening existing Actor machinery in arm B invalidates the run.

Consolidator system prompt (fixed text; the implementer may not soften it):
"You are the consolidator, not the game-playing policy and not a predictor. Read the evidence and the current theory. Produce the smallest revision of the theory that the new evidence warrants. Preserve supported claims. Retract only contradicted claims, recording what retracted them. Keep unresolved alternatives open as competing hypotheses. Mark what is unknown. Never complete the theory merely to make it look coherent. Cite the transition step numbers that support each claim. You cannot act on the game and your output is not a plan."

WORLD_THEORY schema (v1, deliberately boring):
```
WorldTheory:
  version, parent_version
  entities, mechanics, goals_or_objectives, persistent_state
  supported_claims, competing_hypotheses, contradictions, unknowns
  revisions, evidence_refs
```
Every substantive claim payload carries exactly: `claim`, `status` ∈ {supported, tentative, contradicted, unknown}, `evidence_refs`, `counterevidence_refs`, `first_observed`, `last_updated`. No scalar confidence in v1 — it invites numerology before calibration exists. A claim without at least one evidence_ref is invalid and the host rejects the revision.

Consolidation triggers (predeclared):
1. level boundary (levels_completed changes);
2. Actor-invoked `consolidate()` (advertised in the Actor's knowledge_api; bounded);
3. host-detected counterevidence to a declared implication: before execution, the Actor links a supported claim ID/version to an existing experiment ID, typed prediction label and applicability scope. The existing host scorer measures the matching transition. A failed linked typed prediction triggers consolidation; it does not certify semantic contradiction of the prose claim. Without a valid prospective link this trigger is inapplicable; log that boundary. Existing set_experiment remains unchanged; linkage is sidecar metadata;
4. periodic fallback: every 12 executed transitions without any of the above.

Control guarantees (invalid if violated):
- Information parity (USER ACCEPTED: capability parity): equal retrieval capabilities over each arm's own executed prefix, with the same allowed fields/APIs. B sees only its own Actor-retrievable transitions, states and experiment records plus prior accepted WORLD_THEORY, explicitly treatment-derived content. No cross-arm/future evidence or game source. Preserve packets and source file/line/hash, arm/session/cutoff and retrieval provenance. Hashes establish identity; the allowlist/provenance audit establishes access parity. Independent trajectories need not contain equal bytes.
- Evidence immutability: transitions.jsonl and states.jsonl are written only by the existing host writer; the consolidator has no write access to them. Only WORLD_THEORY is revisable, with parent_version lineage.
- No leakage into action: the consolidator has no action authority, no tool that submits, and the Actor is free to ignore the theory. The host never gates an action on theory content.

Budgets (predeclared; the asymmetry IS the treatment and is declared here):
- Actor caps stay symmetric: 129 admitted calls per arm, 200 actions, 600 s model-admission window each.
- Arm B adds a consolidator purse: max 12 consolidator calls per arm-admission, each capped at 4096 completion tokens, wall-bounded 30 s soft / 40 s hard, reasoning off.
- Server chat cap: 259 (as R9) + 12 consolidator = 271 including 1 warmup.
- Wall: serial consolidation has a cumulative 120 s blocking purse, which also caps compensation at +120 s. Count queue wait, packet preparation, model time, validation/publication and cancellation cleanup; no overlapping credit. Stop admission when the purse is exhausted; cap each hard interval by remaining purse. Every attempt (accepted, rejected or timed out) consumes a call; no extra repair request. At 30 s soft stop request further waiting/cancel; hard 40 s refuses late publication and requires terminated work, charged cleanup. Cap breach or unresolved cancellation invalidates. Arm B's actual admission deadline extends by exactly measured blocked time, never more than +120 s. Rationale, stated in the card: the hypothesis under test is whether consolidated theory helps, not whether extra cognition is free; charging consolidator wall against the Actor's 600 s would confound consolidation value with lost play time. Arm A keeps exactly 600 s.
- One private RTX PRO 6000 kernel, internet off, one push, zero retries, session timeout 3600 s, work deadline 3100 s, guard 3500 s. Competition attachment checked from the SDK, not assumed.

Measure (per arm):
- game-progress.jsonl and the official recording, reconciled as in R9.
- world-theory.jsonl: one row per version, with parent lineage, full claim list, and the consolidator request receipts.
- consolidation-events.jsonl: every trigger (which of the four, at which step), packet hashes (proving parity), wall spent, revisions accepted/rejected by host validation.
- For theory survival, the trace table the analysis must produce for every meaningful discovered claim:
  discovered (turn, from set_experiment/rationale/procedure) → entered WORLD_THEORY? (version) → still present after N turns? → revised/retracted after counterevidence? → later relevant (claim cited or mechanically applicable at a later step) → Actor used (claim ID or claim text referenced in rationale/actions/procedure after entry).
- Actor-consumption telemetry, predeclared detection method: string-match of claim IDs and distinctive claim phrases in actions.jsonl rationales and execute_python code after the claim's entry version; also whether the Actor calls `consolidate()` and when.
- Duplicate-probe rate: repeated (action, region) probes yielding no new grid change, per arm; rediscovery latency: steps between a claim's first evidence and its next independent use.

Evidence tiers (predeclared interpretation rules):
- Tier 1 (within-level consolidation): requires no level boundary. Claims: duplicate probes, rediscovery latency, contradiction persistence, survival, retrieval/use.
- Tier 2 (cross-level transfer): requires at least one previously learned mechanic becoming potentially relevant after a level boundary. If no level boundary occurs in either arm, the report MUST say "cross-level hypothesis not tested" and must NOT say "consolidation does not help" — the strongest hypothesis is untested, not refuted. Given R8/R9 (zero levels in six arms), Tier 2 is expected to be untested; that is a finding about tempo, not about consolidation.

Interpretation order: validity first (parity, lineage, no action gating, arm equality), then the survival table, then Tier 1 measures, then Tier 2 if any boundary occurred. Aggregate levels/actions are reported but a null there is not the primary outcome of this card.

Invalid if: arms differ beyond the declared treatment; parity broken; existing machinery removed or weakened in arm B; the theory gates an action; a claim without evidence_refs enters the theory; lineage broken (no parent_version); consolidator caps or wall breached; the theory rendered to the Actor is stale relative to the logged versions; source/notebook drift versus the frozen manifest; unresolved cleanup; competition submission of any kind.

Result map:
- Valid, Tier 1 shows survival plus reduced duplicate probes or reduced rediscovery latency in arm B → one sequential LS20 pair is descriptive evidence consistent with added-compute consolidation benefit; it does not isolate consolidation from extra compute or establish replicated efficacy; next: R11 prospective test (consultation interface, decision leverage) per the sequencing plan.
- Valid, arm B ≈ arm A on all Tier 1 measures AND the survival table shows consolidation failures (claims discovered but not entered, or entered but not used) → consolidation as built adds nothing; diagnose which stage failed before any R11.
- Valid, the Actor never references the theory and rarely consolidates → the interface, not the idea, failed; that is an interface-debt finding, not a refutation (predeclared, per the use-vs-usefulness separation).
- Invalid → preserve, diagnose the first failed boundary, new identity.

Outputs: `analysis/<identity>/` with manifest.json, authorization.json, launch-review.json, CONSUMED.json, launch.json, raw receipts, version-bound downloads of game-progress.jsonl, world-theory.jsonl, consolidation-events.jsonl, both arms' ipc files (by recorded session dirs), RESULT.md separating observation / interpretation / permitted next action, and the theory-survival tables as a first-class artifact.

User decisions recorded: +12 calls at 4096 completion tokens and <=120 s measured extension APPROVED; LS20 selected. One private push per identity conditionally approved only after independent review and Hermes gates, zero automatic retries, no submissions. User also accepted capability parity and linked typed counterevidence in the present implementer conversation. Approvals are unconsumed; NO FREEZE or LAUNCH now. R10b RUN additionally requires Hermes-verified R10a mechanical acceptance.


## Resolved operational contract (independent spec review findings 1–8)

Prediction: in B, at least one newly discovered claim enters an accepted cited version and is referenced by a later Actor request; compared with A, the operational repeated-no-change probe fraction or observed rediscovery latency decreases. This direction is refutable; raw survival/use traces take priority over aggregate gameplay. A string reference alone is not demonstrated causal use. No level or cross-level transfer predicted.

Cap: the numeric Budgets above are binding: 129 incumbent calls per arm (Actor + critic + summarizer), B +12 consolidation attempts, <=4096 tokens each, 30 s soft/40 s hard, cumulative blocked/compensated time <=120 s, 271 server chats including warmup, 200 actions and 600 s Actor eligibility each, one 3600 s GPU reservation. No constructor calls in either arm. Spend cap is GPU wall, no token-billed external provider.

Stop: preparation stops for independent package review, with no freeze or launch. Future execution stops at win, 200 actions, incumbent 129-call purse or Actor admission deadline; consolidator admission stops independently at 12 attempts or 120 s total block. Zero-progress A still permits B in the predeclared paired run. Abort on parity/identity/accounting/lineage/containment/cleanup invalidators; preserve evidence; no automatic retry. No claim/trigger is required to appear; missing opportunities are reported unresolved.

Schema envelope: the exact six-field substantive payload is stored under stable host-managed IDs in `claim_catalog`; WORLD_THEORY's eight semantic/status buckets contain only those IDs, never uncited free text. Version 0 is the empty genesis, first accepted version 1 has parent 0; exactly one accepted parent. Step fields are nonnegative executed-step indices within the named arm/session cutoff. Existing IDs persist for the same claim; proposed rewording requires explicit revision lineage. Changed meaning uses new IDs with replaces/splits/merges lineage. Removal requires recorded counterevidence and a retained contradicted payload, never silent deletion. Revisions are typed lineage records referring to IDs/evidence, not another uncited prose channel. Host assigns new IDs on accepted publication. Validation checks references exist in the supplied immutable prefix; it does not certify entailment.

Triggers: coalesce simultaneous reasons into one request; log all reasons and skips. Count every executed transition including RESET. An attempted consolidation resets the periodic clock; skipped/no-budget attempts do not consume model calls. Actor consolidate() with no new evidence since last attempt is a logged no-op, preventing empty loops. Level boundary has precedence only for display; no reason is dropped. Linked typed counterevidence is logged against its prospective claim version, experiment, prediction and executed successor; it is not an automatic retraction rule.

Context freshness: publish accepted revisions serially; rejected revisions leave the last accepted theory active. At Actor request assembly snapshot and log exact accepted version/digest, context text and request ID. Normal in-flight Actor requests retain their assembly-time snapshot; only a stale snapshot at assembly is invalid.

Two evidence tiers remain binding. Survival horizons N = 1, 3 and 12 executed transitions after entry; beyond episode termination is right-censored, never failure. Trace every discovered meaningful claim in both arms using the same human-reviewed discovery/use rubric from rationale, experiment and procedure records. B additionally tracks theory versions. Mechanical ID/distinctive-phrase matches are `referenced`; reviewed contextual evidence can label `used`; unsupported applicability remains unclassified. Record first observation, entry/revision/retraction, each survival horizon, later relevance and use with raw citations. Rediscovery latency is steps from first evidence to first later independent supported use; unreused claims are right-censored. Distinguish first post-entry reference latency from rediscovery.

Duplicate-probe proxy: key = (level count, exact action including CLICK x/y or whole-grid region for non-CLICK, pre-grid digest). Numerator: repeated identical key with unchanged before/after grid; denominator: all later occurrences of previously observed keys. Undefined when denominator zero. Report counts and sequences in both arms. No-change repetition may still provide information; do not equate this proxy with proven waste.

Outputs additionally include exact source-prefix packets/receipts, scoped evidence catalog, linkage records, context snapshots, raw elapsed intervals and deadline credit records, a per-claim survival table including right-censoring and control-arm opportunities, and a prelaunch implementation checklist. A local schema or transport fixture is not remote qualification or a scientific outcome. If no qualifying level boundary occurs: **cross-level hypothesis not tested**.
