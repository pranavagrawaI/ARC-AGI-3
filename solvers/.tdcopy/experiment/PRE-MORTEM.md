# Pre-mortem — 2026-10-04

Failure scene: three days from now the ten-GPU-hour allowance is spent. A score table and convincing traces suggest the hybrid is better, yet we cannot tell whether executable construction and repair helped. A timing or measurement difference may have driven the result, and we choose the wrong next research direction.

Objective: equal-compute score improvement plus a supported mechanism explanation. Same model, small game set, voluntary construction, score decisive. Nine GPU-hours remain under conservative accounting. Evidence examined: PLAN.md, COMPARISON-DRAFT.md, q4 raw receipts, hybrid/canonical/donor call chains and current local tests.

## 1. We test delegation and call it construction efficacy

- **Chain:** remove invoke_builder or switch TYCHO_MODE -> other model facilities remain, or ownership/seeding/feedback change together -> the advertised single-mechanism contrast is false -> a score delta receives the wrong explanation.
- **Evidence:** observed mode/agent code couples several behaviors. Orchestrator prohibits Actor writes to canonical world_model.py, but native Python and notes can still implement dynamics computations. No ablation is built yet.
- **Risk:** high likelihood unless defined carefully; high impact; poorly detectable from aggregate scores; cheap recovery before launch, expensive afterward.
- **First warning/probe:** inspect the complete proposed ablation diff and actual tool/workspace behavior. A mode-name change is insufficient evidence of a narrow intervention.
- **Response:** define the intervention operationally from discovery traces, then freeze a fresh matched pair. Distinguish delegation, canonical model support and feedback. Accept that one affordable contrast may identify only a bundle.

## 2. The mechanism is silently absent

- **Chain:** Actor declines construction or requests it late -> no exposure or budget rejection -> zero Builder calls is interpreted as ineffectiveness -> an inert ablation consumes the attribution budget.
- **Evidence:** observed q4 SB26 request at sequence 15; a 164.423-second decision could not clear the 300-second Actor reserve plus 5 seconds for return. Old ordinary allocation also expired 240 seconds after decision start. Future voluntary uptake is unknown.
- **Risk:** high; high attribution impact; easy detection with the right traces; partial recovery within budget.
- **First warning/probe:** count tool offered, requested, admitted and completed separately; reproduce late/short requests.
- **Response:** v3 allocates at invocation, capped at 240 seconds, retaining min(120 seconds, half the usable remainder) for Actor and 5 seconds for return. Local short-budget construction and late-request arithmetic pass. Deeper qualification removes the 16-action ceiling. Do not force construction or require a voluntary invocation for a performance-run pass. Non-use leaves conditional efficacy unidentified.

## 3. Execution masquerades as correct reasoning

- **Chain:** model passes its interface check, tool status says complete, or retrospective prose sounds right -> proxies are scored as mechanism correctness/recovery -> selection rewards unsupported reasoning.
- **Evidence:** q4 Python tracebacks appeared inside complete tool records. An earlier local fixture won despite rejected Builder calls. q4 first-shot game-mechanism correctness is unscored.
- **Risk:** high; high impact; difficult detection from summaries; recoverable if raw commitments survive.
- **First warning/probe:** an alleged prediction lacks pre-observation bytes/timing, or recovery fixes a NameError rather than a falsified game law.
- **Response:** retain pre-observation claim/model hash, action, expected observable predicate and subsequent observation. Separate construction, executable acceptance, predictive support and use in action selection. One matched transition supports that prediction, not an entire law. Report unmeasured claims and coverage denominators. Recovery runs from observed contradiction to a subsequently verified correction; unfinished recovery is censored. Do not add observability prompts to only one arm.

## 4. Equal configured time hides unequal work

- **Chain:** write/read get separate timeouts, tool child survives cutoff, or readiness differs -> extra computation or late actions -> the score gap is attributed to reasoning.
- **Evidence:** local IPC reproduction took 1.701 seconds under a 1-second allowance; a child survived worker close. Both are now fixed/tested. q4 startup consumed substantial time. Three-game tests do not establish 25-game scheduler fairness.
- **Risk:** medium after fixes; high impact; hard detection from score alone; costly recovery after execution.
- **First warning/probe:** out-of-window action timestamps, live child processes, readiness differences or missing interrupted-call accounting.
- **Response:** common allocation/game caps, exact timing, one IPC deadline and dedicated worker process-group cleanup. Local reaping does not prove SGLang cancelled provider computation; retain that uncertainty. Compare allocated-time scores; use post-readiness curves diagnostically. Real engineering overhead belongs in the whole-system contrast, which is not a pure reasoning contrast.

## 5. Three games produce a persuasive accident

- **Chain:** small inspected set plus stochastic generations -> one lucky game dominates -> screening becomes a general architecture claim -> research effort moves to the wrong lever.
- **Evidence:** one shallow q4 pass; deeper v3 exposure, effect size and variance remain unknown.
- **Risk:** high for overinterpretation; high impact; poor detection from one pair; partial recovery through replication.
- **First warning/probe:** mean-only reporting, one game accounting for the gain, or changing the game list after outcomes.
- **Response:** fixed game list, randomized arm order, per-game deltas/time curves and fresh mechanism comparison or targeted replication. Accept a screening result and possible inconclusiveness. No public-25/hidden-set superiority claim follows.

## 6. Discovery consumes the attribution budget

- **Chain:** extra smoke runs, startup and retries -> only one mechanism arm fits -> we answer whether the bundle looks promising but miss the user's required mechanism question.
- **Evidence:** one hour charged conservatively; q4 queued roughly 95 minutes. The stale draft still specified public25 and allocations inconsistent with the latest agreement.
- **Risk:** medium-high; high impact; early ledger detection; poor recovery at the cap/calendar deadline.
- **First warning/probe:** next launch leaves insufficient budget and calendar time for both sides of the discriminating comparison.
- **Response:** revised envelope: 1 hour consumed, 1 for qualification, 3 for whole-system pair, 3 protected for conditional mechanism pair, 2 contingency. Reserve a complete pair before its first arm. A justified second qualification hour uses contingency. No automatic retries. If construction is unused, skip the inert ablation and explicitly choose depth/replication or leave budget unspent.

## 7. Evidence never reaches the decision

- **Chain:** huge virtualenv/output inventory or receipt-write failure -> collector stalls or cleanup fails -> summary appears complete but decisive evidence is missing -> next launch proceeds without diagnosis.
- **Evidence:** q4's initial collector timed out; selected traces arrived later. An injected artifact-write failure stranded worker bookkeeping/admission. v3 finalization now releases resources on that failure. All 55 q4 provider requests were eventually checked: no orphan tool replies or broken user boundaries.
- **Risk:** medium; high impact; detectable if collection has its own completion criterion; recoverable while remote output persists.
- **First warning/probe:** benchmark exists but per-game action/role/exit receipts do not; rehearse retrieval and the audit on preserved outputs.
- **Response:** venv moved to /tmp; retrieve benchmark/config/action/role/exit receipts before bulk blobs. Inspect results before follow-up. A read-only collector cannot replace an agent's interpretation. Missing evidence is pending, never zero. Summary-first collection still needs implementation before the next launch; it is not claimed fixed.

Launch-preparation follow-up: summary-first collection is now implemented and rehearsed on 34 actual q4 artifacts. New runs emit a hash-bound inventory, so collection avoids dependency-file enumeration. Byte equality and all three action/token ledgers reconcile. See REVIEW.md's launch update for the additional independent-review fixes and tested watchdog. This records the changed state without treating the earlier pre-mortem as a prediction of success.

## 8. Fixes invalidate old qualification

- **Chain:** legitimate repair changes budgets/policy/reporting -> old PASS is carried forward despite new bytes -> a new defect or confound enters the pair.
- **Evidence:** v3 allocation is a policy change. Added game-name reporting broke ExampleGame; the runner test caught it, its failure log was preserved, and the fallback was fixed. v2 is separately snapshotted with hashes.
- **Risk:** medium; high impact; detectable with artifact-specific checks; cheap local recovery.
- **First warning/probe:** source hash differs from qualified payload, or a newly touched path has only static checks.
- **Response:** keep q4 immutable; label v3, rebuild/hash/check generated artifacts and real worker/runner paths. Use a distinct hash-bound source dataset. Freeze the same candidate in both mechanism arms except the declared intervention. Local success does not establish live GPU compatibility or efficacy.

## Decision and remaining exposure

Proceed toward one deeper qualification once current local checks and evidence collection are ready. Do not force construction. Freeze/review both arms and their resource semantics before whole-system comparison; define the exact intervention before claiming mechanism attribution.

Examined: objective, success test, budget/calendar, optional activation, mode semantics, deadlines, cleanup, scheduling interface, provider requests, score/usage evidence, collection and provenance. Unestablished: live v3 hardened execution, server-side cancellation, voluntary uptake, variance, future ablation semantics and generalization. This is not a general sandbox-security or serving-stack audit.

Changes worth making now: retain supported execution fixes, correct small-set/resource instructions, collect decisive artifacts first, and define measurement/ablation boundaries before spending the reserved budget. Consciously accept limited statistical precision, uncertain uptake and an inconclusive result. Do not manufacture certainty by forcing activation, interpreting non-use as failure or exceeding the cap.
