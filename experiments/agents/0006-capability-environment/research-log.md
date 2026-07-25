# Research log: 0006 capability environment

## 2026-07-25 — reusable 0002–0005 substrate and authority boundaries

Question: which existing mechanisms can 0006 v0 reuse without changing their
claim, which ideas require a new interface, and what do the existing artifacts
actually license?

### Evidence grades

The repository has three materially different evidence grades. They must not be
collapsed into one lineage:

1. **Current main, reusable implementation.** `main` contains 0002 and the frozen
   0003 negative control. The current 0002 and 0003 suites pass together: 70
   tests via
   `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python3 -m unittest experiments.agents.tests.test_llm_repl_agent experiments.agents.tests.test_reconstruction_uptake`.
2. **Current main, invalidated negative control.** 0003 is deliberately preserved,
   not repaired. Its licensed result is only L0 executable API compliance because
   an action echo and shared mutable state can satisfy its checker
   (`experiments/agents/0003-reconstruction-uptake/INVALIDATED.md`, “Licensed
   result”; `reconstruction_uptake/compliance.py::assess_reconstruction_compliance`;
   `reconstruction_uptake/model_policy.py::ModelReplayPolicy._invoke_artifact`).
3. **Archive-only design evidence.** The counterexample ledger, model-backed
   replay, and Gemma compatibility code are not on `main`; they live on
   `archive/invalidated-freeze-chain-20260724`. They are useful design evidence,
   not code 0006 may import unchanged. Every archive citation below is qualified
   with that ref. Restoring any part changes the current source set and requires
   a new 0006-owned seam, tests, source closure, and acceptance evidence.

The per-experiment `research-log.md` pattern is the repository convention for
research notes (`experiments/agents/0000-random-agent/research-log.md`,
`0001-ls20-observation-option-agent/research-log.md`, and
`0002-llm-repl-agent/research-log.md`). This note follows it for 0006.

### Reuse unchanged from current main

| Mechanism | What 0006 v0 can reuse unchanged | Boundary that must remain explicit |
|---|---|---|
| Generic persistent Python runtime | Reuse `llm_repl_agent.workspace::PythonWorkspace`, `PythonResult.model_text`, the NumPy/Pillow allowlist, bounded output, timeout/restart behavior, bubblewrap mount construction, and seccomp/audit-hook restrictions (`workspace.py:29-101, 186-225, 285-448`). | It is a research sandbox, not a formally verified hostile-code boundary. It gives the model a writable persistent workspace; it must not become the authoritative evidence or belief store. |
| Sandbox audit | Reuse `sandbox_admissibility::run_sandbox_admissibility_suite` and its source hash as a pre-live interface check (`sandbox_admissibility.py:13-53`; adversarial probes begin at lines 62-180). | Passing the enumerated probes licenses only that exact runtime/source/environment. It does not prove absence of native-runtime, resource, timing, or same-user attacks. |
| Public observation encoding | Reuse pure transforms `observations::observation_metadata`, `pixels_to_png`, `representation_descriptor`, and `ColorPermutation` (`observations.py:28-147`). | `ObservationStore` writes its projection into the model-writable workspace (`ObservationStore.write_observation`, lines 178-208). Those files are convenient model inputs, not immutable host evidence. |
| Model transport and wire commitments | Reuse `client::serialize_request_payload`, `ChatResponse`, and `OpenAICompatibleChatClient`; reuse the exact pre-redaction request and raw-response commitments produced by `process_log::ProcessLog.write` (`client.py:20-204`; `process_log.py:48-115`). | Transport identity and byte commitments establish what was sent and received. They do not establish provider determinism, semantic correctness, or epistemic uptake. |
| Run audit closure | Reuse `evidence::evidence_hashes` and `validate_run_evidence`, including non-following treatment of contestant symlinks and exact request/response archive validation (`evidence.py:34-95, 98-218`). Reuse `PythonWorkspace.evidence_state` for model-authored workspace deltas (`workspace.py:484-516`). | This is end-of-run tamper evidence and provenance. It is not an online evidence ledger, claim ledger, or reconciliation gate. |
| Bounded recent transcript | Reuse `conversation::RollingConversation` as the recent-turn cache (`conversation.py:15-40`). | It evicts whole turns by count only. It is not the 0006 Context Assembler because it has no active-claim, contradiction, retrieval, or epistemic-status semantics. |
| Final action syntax | Reuse `agent::parse_decision_message` for the final legal-action/data projection (`agent.py:116-150`). | In 0002, `LlmReplAgent.act` can return a legal `Decision` after syntax repair alone (`agent.py:196-314`). In 0006 it must run only after the new decision protocol admits the commitment; legality is not epistemic authorization. |

The clean reuse strategy is composition: 0006 should consume these current-main
modules as leaf infrastructure and add its own controller around them. Copying
the archived 0004/0005 tree back into main would import obsolete claims and
authority assumptions together with the useful mechanics.

### New seams required for 0006 v0

1. **Host-owned immutable evidence archive and read-only projection.** Keep
   authoritative observations/actions outside `PythonWorkspace.root`; expose
   copies or deterministic query results into the workspace. `ObservationStore`
   is reusable as a projection writer, but cannot be the authority because the
   model can overwrite its files. The archive design usefully separated raw from
   derived records and chained their bytes
   (`archive/invalidated-freeze-chain-20260724:experiments/agents/0004-counterexample-gated-ledger/counterexample_ledger/evidence.py::EvidenceLedger.append_raw_observation`,
   `append_raw_action`, `append_derived`, and `_load_and_verify`, lines 18-251).
   Reimplement and revalidate that separation under 0006 rather than importing
   the archive.

2. **Evidence-operator registry.** 0002 exposes frames and history but has no
   typed callable boundary saying “deterministic measurement over immutable
   evidence, no semantic or belief authority.” 0006 needs operator identity,
   version/hash, input evidence IDs, canonical output, and an audit event. The
   archived derived-evidence extractor identity is a useful record shape
   (`EvidenceLedger.append_derived`, lines 85-129), not a complete operator
   interface.

3. **Model-authored epistemic ledger.** Nothing on current main provides stable
   claim IDs, statuses, counterevidence links, or revision/supersession history.
   Archived `EvidenceLedger` stores host raw/derived evidence, not model-owned
   beliefs. 0006 therefore needs a separate append-only `EpistemicLedger`; the
   host validates schema and history while the model retains semantic and
   revision authority.

4. **Decision protocol before environment authority.** Insert a gate between
   model response and the `Decision` returned from `LlmReplAgent.act`, and a
   post-successor check between `LlmReplAgent.observe` and the next `act`
   (`agent.py::LlmReplAgent.act`, lines 196-314;
   `agent.py::LlmReplAgent.observe`, lines 413-426). A commitment must bind the
   proposed action, purpose, relied-on claim IDs, and model-selected typed
   observable assertions. The archived `DecisionController.admit` proves the
   value of exact checkpoint/candidate/action/evidence binding
   (`archive/.../counterexample_ledger/decisions.py::DecisionController.admit`,
   lines 187-288), but its mandatory executable candidate set and
   hypothesis-discrimination policy are reconstruction-experiment assumptions,
   not generic 0006 substrate.

5. **Typed observable assertions.** Do not inherit the archive's one
   `committed_prediction` object or its T0 fingerprint ontology. 0006 needs an
   extensible, game-general assertion registry over permitted public
   before/after fields, with assertion-specific mechanical evaluators and
   explanatory prose kept non-authoritative. The archive's exact
   legal-action coverage and candidate hashes
   (`counterexample_ledger/decisions.py::CandidateSet.create`, lines 28-64)
   remain optional executable-reconstruction instrumentation, not a prerequisite
   for every action.

6. **Reconciliation gate with disposition, not forced theory change.** Archived
   `DecisionController.reconcile` creates a counterexample on digest mismatch,
   then `record_revision` requires at least one hypothesis to be revised or
   rejected (`archive/.../decisions.py`, lines 290-339). That is too strong for
   0006: a mismatch may be an assertion/grounding error, an unresolved anomaly,
   or compatible with retaining the relied-on claims. The new gate should block
   the next environment action until every mismatch receives a model-authored
   disposition, without the host choosing the diagnosis.

7. **Context Assembler.** Compose the next bounded packet from current
   observation, active claim records, unresolved mismatches, recent continuity,
   and explicitly retrieved immutable evidence. `RollingConversation` remains
   the recent-continuity component only. Assembly decisions and omissions need
   audit events so compaction cannot silently promote a belief to fact.

8. **0006 audit schema and execution authority.** Emit structured events for
   evidence reads/operator calls, claim creation/revision, commitment,
   assertion evaluation, mismatch disposition, assembled context, and final
   admission through current `ProcessLog`. The existing experiment-spend
   authority code validates exact pilot/freeze artifacts and atomically consumes
   pair grants (`authority.py::validate_pilot_approval`,
   `consume_pair_authority`, `validate_final_freeze`, lines 270-383, 406-458,
   746-835). Its schemas are 0002 experiment-specific; 0006 needs new authority
   artifacts that separately cap model/network/environment actions and bind the
   new capability manifest.

### Archive-only ideas worth re-deriving, not reusing unchanged

- **Hash-closed executable bundles.**
  `archive/.../counterexample_ledger/bundle.py::BundleSnapshot.capture` and
  `create_bundle` bind source, state, schema, prompt, configuration, extractor,
  runtime, parent, and update commitment (lines 52-219). This is valuable if
  0006 later admits external executable conjectures, but it is not necessary for
  the v0 epistemic/decision substrate.
- **Fresh candidate isolation.**
  `archive/.../counterexample_ledger/isolation.py::IsolatedCandidateEvaluator`
  creates one current-0002 `PythonWorkspace` per legal action and binds child
  state to the frozen parent (lines 58-193). It solves cross-candidate mutable
  state for that experiment; it would destroy the persistent generic workspace
  if made the universal 0006 execution model.
- **Prospective replay ordering.**
  `archive/.../counterexample_ledger/replay.py::SequentialGoldenReplay` enforces
  `ready → committed → revealed → updated` and makes failed update terminal
  (lines 19-154). Re-derive the ordering for live 0006 action commitments, but
  do not import an offline replay tied to a recorded action.
- **Mechanical claim limits.**
  `archive/.../counterexample_ledger/claims.py::issue_claims` correctly makes
  L0–L5 conjunctive and prohibits “reconstruction” below L2+T2 and “world model”
  below L5 (lines 28-92). Preserve the discipline, not this experiment-specific
  ladder.
- **Rendered model contract and one-shot execution controls.**
  Archived 0004 model replay separated bootstrap, decision, and revision
  parsing (`model_backed_replay/responses.py::parse_bootstrap_response`,
  `parse_decision_response`, `parse_revision_response`) and used a transport
  that rejected calls in render-only generations
  (`model_backed_replay/transport.py::RenderOnlyTransport`). Archived 0005 added
  exact one-shot claim ownership and intention-to-treat response journaling
  (`run_reasoning_off_compatibility_probe_v7.py::_open_consumption_claim`,
  `_persist_owned_consumption_claim`, `_persist_reasoning_off_terminal_report`).
  These are useful patterns, but they are archive-specific launchers with
  model/provider/fixture assumptions. Current 0002 transport plus a new
  0006-owned authority schema is the smaller path.

### What the evidence does not license

- **0002:** its tests license the current generic runtime, serialization,
  transport, sandbox probes, logging, and budget behavior. They do not establish
  a secure hostile-code sandbox, durable belief persistence, calibrated
  prediction, revision quality, or gameplay competence. The research log's
  recorded smoke/probe runs completed no level and explicitly treats paired
  observations as single stochastic comparisons
  (`0002-llm-repl-agent/research-log.md`, 2026-07-13 entries).
- **0003:** licenses L0 executable API compliance only—no action-conditioned
  prediction, reconstruction, learning, planning adequacy, or gameplay benefit
  (`0003-reconstruction-uptake/INVALIDATED.md`, “Licensed result” and “Freeze
  rule”). Its commitment-before-reveal shape is useful negative-control
  evidence, not an accepted substrate.
- **Archived deterministic 0004:** the frozen artifact's own
  `freeze_scope.establishes` is only deterministic evidence/control plumbing,
  exact source-to-golden projection, and a T0 development-window smoke seam.
  Its `freeze_scope.does_not_establish` explicitly excludes transcript-eviction
  persistence, LLM hypothesis formation/revision, held-out or decision
  improvement, gameplay adequacy, and hostile-code security
  (`archive/.../0004-counterexample-gated-ledger/experiment/deterministic-freeze.json`,
  keys `freeze_scope`, `status=deterministic_substrate_frozen_model_unvalidated`).
  The 6/13 nearest-retrieval result gains only at inspected step 17; 13/13 public
  status equals a trivial no-change predictor
  (`archive/.../0004-counterexample-gated-ledger/README.md`, “Verified offline
  smoke result”; `offline.py::analyze_retrieval_gain` and
  `analyze_public_status`). It is not reconstruction or transfer evidence.
- **Archived model-backed 0004:** dry-run v9 rendered fixture bootstrap,
  decision, revision, eviction, and report plumbing under zero model/network/
  environment authority. The fixture is intentionally wrong. Its README
  explicitly leaves real model bundle production, belief persistence,
  meaningful counterexample revision, changed later predictions, T0
  improvement, token fit, provider identity, latency, and cost untested
  (`archive/.../0004-model-backed-offline-replay/README.md`, “What the dry run
  establishes” and “What remains untested”;
  `model_backed_replay/orchestration.py::build_dry_run`).
- **Archived 0005:** it establishes bounded compatibility/transport observations
  for specific Gemma/OpenRouter generations, not a capable belief-revision
  agent. The v6 predecessor consumed its 2,048 completion-token ceiling with
  `content=null`; v7 was only an awaiting-approval reasoning-off bootstrap probe
  (`archive/.../0005-gemma-model-backed-offline-replay/approval-requests/compatibility-probe-v7.json`,
  keys `predecessor_probe.result`, `requested_stage`, `status`, and
  `anti_actions`). No 0005 artifact licenses full offline replay, decision or
  revision requests, environment actions, live play, benefit, or generality.

### Decision

Build 0006 v0 as a thin new controller over current-main 0002 infrastructure.
Reuse the generic workspace, pure public-observation transforms, transport,
wire commitments, run-evidence closure, recent-turn cache, and final action
parser unchanged. Add six 0006-owned seams: immutable evidence plus operators,
model-authored epistemic ledger, typed action commitment/assertions,
model-authored reconciliation dispositions, context assembly, and a new
capability/authority manifest.

Do not restore or import archived 0004/0005 modules. Treat their hash chaining,
ordering, isolation, claim limits, and one-shot controls as adversarial design
cases to reimplement narrowly and revalidate against 0006's semantics. This
preserves the useful decisions without inheriting an invalidated freeze chain
or overstating what its fixture and compatibility evidence proved.
