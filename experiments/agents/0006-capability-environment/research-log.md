# Research log: 0006 capability environment

## 2026-07-26 — issue #9 built as four prototypes; three substrate gaps found

Question: can the strong generic 0002-R control and a provisional treatment be
built from one factory, and what does building it reveal about the invariants
issue #9 assumed?

Built in `p9/` as four small prototypes rather than a harness rewrite. Nothing
under `llm_repl_agent/` or `experiments/harness/{runner,trace,types}.py` is
modified — all of it is inside `participating_source_hashes()`, so an edit would
invalidate the 0002 release gate. Every addition composes over those modules.

### Gaps the candidate exposed

Three of #9's interface-quality invariants did not hold on unmodified 0002, and
none was visible from prose:

1. **Authoritative episode data was model-writable.** `ObservationStore` writes
   the public observations into `PythonWorkspace.root`, so the model can truncate
   the only copy. Fixed by an append-only host-owned archive bound
   read-only at `episode/` (`episode_archive.py`, `projected_workspace.py`).
   P9.2 has the fixture model destroy the workspace copies first, then recover
   every evicted step from `episode/`, whose write attempt returns `EROFS`.
2. **Transcript eviction was silent to the model.** 0002 logs `context_eviction`
   host-side only. Fixed by a pointer-only notice naming the dropped steps and
   the authoritative paths, generated from a fixed template so the notice is a
   pure function of the evicted step set (`conversation.py`).
3. **Transport retries were unreconstructable.** `OpenAICompatibleChatClient`
   retries transient failures with no record, so two requests for one
   deliberation round left one trace. Fixed by moving the retry loop into
   `transport.py` with the inner client at `rate_limit_retries=0`.

Two more are documented rather than fixed, because the code creating them is
frozen. Both were observed defeating a model, not derived from reading the source:

- **`sys.path`.** The worker runs `python3 -I`, so a model-written module is not
  importable without `sys.path.insert(0, ".")`, and nothing said so.
- **Observation filename width.** The model is shown only the glob
  `observations/step-*.json`. The zero-padding is hidden: concrete filenames reach
  it solely through `recent_transitions`, which is empty at step 0 and carries the
  last eight steps. Models construct `step-%04d.json` and get `FileNotFoundError`.
  Now stated with a worked example and **pinned against the filenames
  `ObservationStore` actually writes**, so the contract cannot drift from the code;
  two mutants guard it.

The pattern is worth naming: every path the model must *construct* rather than
copy is a rediscovery cost, and a glob in the contract hides exactly the part that
has to be guessed.

### What the prototypes establish

- **P9.0.** 95 of 104 manifest paths and 84/85 and 94/95 observed
  request-payload paths are identical between arms; every difference is declared
  by `sentinel_probe_note`. A third arm with one smuggled prompt sentence is
  rejected on `model_visible.system_prompt.sha256`.
- **P9.1.** A helper module and notebook authored at step 0 are reused at step 1
  against an observation that did not exist when they were written.
- **P9.2.** Above.
- **P9.3.** Five tool-failure signatures are distinct; syntax and runtime
  failures share a kind and are separated by the exception named in the
  traceback. Timeout restarts the interpreter, loses in-memory state, keeps
  files and the read-only record, and reports all three. The illegal action
  returns to the model; the retryable transport failure is retried once and is
  absent from every request the model saw; the fatal one is not retried.

### Deleted after measurement: the executed-transition log

`episode/history.jsonl` was built, then removed once existing runs were queried.
Across 8 recorded 0002 runs (414 actions, 165 tool calls, 38-87 evictions each):
`public_history.jsonl` was read **0 times**, though it is named in the system
prompt and in every request's `full_history_file` field; `observations/` drew 405
explicit `step-NNNNN` references but **96% of them point inside the transcript
window**, only 4% past it. The same records are already injected inline as
`recent_transitions` for the last 8 steps, so the file's sole unique content was
the action taken more than 8 steps back. The agent never worked at that horizon.

That null licenses "the file adds nothing to an agent operating inside an 8-step
horizon" — not "long-horizon action history is worthless", which nothing tested.
Deleting it is nonetheless the right call: the unique content is exactly what a
0006 treatment should earn rather than inherit, and re-adding it is now a
declared capability guarded by the
`transition-log-reintroduced-into-the-substrate` mutant.

Two things the deletion taught:

1. **It cost prompt rather than saving it, until the advice came out.** Stating
   that no action record exists ran longer than describing the file, and
   "keep your own notes if you want a longer record" is host-authored policy of
   exactly the kind the contract forbids. Cutting both took the payload delta
   from +12.4%/+15.2% to +10.8%/+13.3%.
2. **The arm verifier catches divergence, not drift.** A surface added to *both*
   arms passes cleanly, so a re-added transition log was invisible to it. The
   shared episode contract is now pinned against a literal in P9.0
   (`FROZEN_EPISODE_CONTRACT`).

### Removed after the escalation: the episode archive

The read-only `episode/` archive is gone, and with it `episode_archive.py` and
`projected_workspace.py`. It satisfied #9's "authoritative episode data"
invariant and was the most carefully engineered part of the layer. It went
because the escalation forced the question its own checks could not put:

- **Availability was never the gap.** `ObservationStore` keeps every observation
  in the workspace for the whole run; eviction does not delete them. The 17
  post-eviction reads in recorded runs were served fine by unmodified 0002.
- **Authority had no measured demand.** No recorded run shows a model corrupting
  its own observations. The archive defended against an unobserved threat.
- **P9.4 declined to defend it.** Pruning the archive did not fail the task,
  because the policy falls back to the workspace copies. The task verifier was
  telling the truth where the contract verifier could only restate the
  requirement.

Cost recovered: the prompt addendum fell from +816 to +532 characters, and the
payload delta from +12.4%/+15.2% to **+5.5%/+7.6%**. 0002-R is now two additions
over unmodified 0002: a pointer-only omission notice and a logged transport
retry.

**This is a stated deviation, not a silent drop.** `CONTRACT.md` records that
0002-R does not provide authoritative episode data, and what that forecloses: a
0006 claim of the form "the model reconstructed X from its record" cannot be
defended against "the model wrote that record itself" on this substrate. If 0006
needs the separation it must register the archive as a declared treatment
capability, where the arm diff will show it doing work. P9.2 now measures the
accepted risk — the observation files are model-writable and nothing restores
them — rather than asserting it away.

### Testing the tests

Passing self-verifying prototypes establish nothing on their own, so
`p9/mutation_probe.py` breaks one named property at a time and requires
something to notice. Its first run exposed two vacuous claims:

- **`EpisodeArchive.verify()` was never shown able to fail.** P9.1 and P9.2 both
  asserted that the archive verifies; a `verify()` hardcoded to report a match
  passed every prototype and every test. The tamper-detection claim was empty.
  Fixed by direct archive tests that corrupt and delete observations and require
  the mismatched record to be named.
- **P9.3 read `workspace_restarted` off the raw tool record, not the
  classifier**, so a classifier that stopped reporting restarts survived the
  prototype. P9.3 now asserts through `classify_tool_result`.

After the fix: 12/12 mutants caught, sources verified restored
(`p9/artifacts/p9-mutation-probe/results.json`). The prototypes exercise
properties end to end; the unit tests carry the negative half an end-to-end run
cannot reach.

Frozen contract: `p9/CONTRACT.md`. Evidence: `p9/artifacts/*/evidence.json`.
Tests: `experiments.agents.tests.test_p9_shared_arm` (31 tests).

### Delete-centric pass

Four overdeletion passes across four axes (dead code, mechanisms, requirements,
whole prototypes) with the suite as the verifier. Six deletions survived and were
kept; the failure boundary was crossed nine times. Two results were worth more
than the code they removed:

1. **A fake deletion nearly licensed removing half the verifier.** The first
   attempt at cutting P9.0's observed request-payload comparison only *inserted*
   a line — it never removed the loop — and reported green. Redone properly it
   was still green, which meant the verifier could not see its own strongest half
   go missing. Strengthened rather than deleted: a mutant now diverges the arms
   on a payload path the manifest has no entry for, and only the observed
   comparison catches it.
2. **The probe scored a stale patch as a survivor.** After the chain digest was
   deleted, `archive-verify-always-reports-match` no longer matched any source
   text and was silently counted as surviving. A probe that stops testing
   something must fail loudly, not report a gap it is no longer measuring.

`p9_1_cumulative_loop.py` looked deletable against the mutation probe. The
escalation below overturned that: it catches an archive-pruning regression P9.4
misses. It stays.

### Escalation: a verifier that measures the harness, not the contract

The delete-centric pass ended on a circularity. Requirement-level deletions all
failed, but only because the prototypes' checks encode the requirements — a
contract verifier can confirm a contract and nothing else. P9.4 replaces the
success criterion: a reactive policy (it reads the conversation and decides from
the last tool result, so it can be defeated) maintains an index of every
observation seen, and the host grades the finished index against digests it
computed itself. Seven conditions — clean, eviction, interpreter timeout,
retryable transport failure, oversized output, malformed tool arguments, all at
once — each complete correctly, and each asserts its fault actually fired.

The task reads whichever observation directory exists, workspace or archive. That
is deliberate: it measures whether the model can reach every past observation, not
whether the archive exists. A task encoding the second would defend the
requirement rather than test it.

**P9.4 is orthogonal, not strictly stronger.** Under a mutant wiping
model-authored files between actions, P9.1, P9.2 and P9.4 all fail. Under a mutant
pruning the archive to its two newest observations, P9.1 and P9.2 fail and P9.4
does not — it falls back to the workspace copies. This also answers the earlier
REQUIRE question: **P9.1 is not redundant** and stays. An initial reading that it
caught nothing unique came from a mutant tailored to P9.4's own artifact.

**What the escalation bought.** Making the archive write an observation with
different key ordering — so the model's two model-visible paths to one public fact
return different bytes — passed every prototype, every unit test and every mutant.
That is a defect the model could not detect either: two incompatible versions of
one public fact, no way to tell which is authoritative. P9.4 now compares both
projections per condition, and `observation-projections-disagree` guards the
check. Final: 14/14 mutants caught.

### The verifier's own defects

Running the probe adversarially found more wrong with the verifier than with the
code under test. Three, in order of severity:

1. **It overwrote the evidence it exists to protect.** The probe runs prototypes
   against deliberately broken source, and `Checks.finish` writes `evidence.json`
   before raising, so every probe run replaced the committed artifacts with mutant
   output. `p9-2-context-loss/evidence.json` sat on disk claiming `passed=false`.
   Committing after any probe run would have shipped evidence asserting a
   prototype fails. Prototypes now honour `P9_ARTIFACTS_DIR`; the probe redirects
   to scratch.
2. **A stale patch counted as a survivor.** A mutant whose target text no longer
   existed was scored as surviving — understating coverage while hiding that the
   probe had stopped testing anything. Stale is now a hard failure.
3. **No concurrency lock.** Two runs restore from each other's stale snapshots and
   can leave mutated source behind. It now refuses to start while another holds
   `.mutation-probe.lock`.

The general lesson is the one worth keeping: an instrument that writes to the
same path it measures will eventually report on itself. None of these were
visible from reading the code; all three appeared only by running the verifier
against a system it was supposed to condemn.

### What this does not license

No gameplay, learning, or 0006-intervention claim: the model is a fixed script
and the environment is a recording. No claim that the manifest covers every
model↔harness crossing — the verifier's reach is the manifest plus the recorded
payloads. No live endpoint, budget authority, release gate, or RHAE readiness.
No security claim: the read-only bind constrains the sandboxed model, it does
not make the sandbox a hostile-code boundary.

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
