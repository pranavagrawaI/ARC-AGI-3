# Builder CodeAct architecture

**Normative protocol:**
[`BUILDER_PROPOSAL_PROTOCOL.md`](../solvers/tycho_nooa/BUILDER_PROPOSAL_PROTOCOL.md).
This document records the architectural evidence and experiment rationale; the protocol
owns value schemas, hashing, lineage, evaluator determinism, publication, and proof
obligations.

**Revised judgment:** use CodeAct as a pure proposal machine. Do not apply
`CodeActStrategy` directly to the current `FocusedBuilder`, and do not let `evaluate()`
mutate a transaction working incumbent. Keep the fresh Builder agent, trigger policy,
frozen evidence frontier, verifier, transaction, publication gate, and advisory handoff.
Replace the model-facing native-tool construction loop with a CodeAct session inside a
disposable construction realm. Its only egress is a typed candidate submission or final
report. The host evaluator returns immutable receipts; only the host transaction later
interprets those receipts as retention or publication decisions.

```text
Builder computes
Host judges
Transaction remembers
Actor acts
```

This is an architecture hypothesis until a matched zero-ARC replay shows that it produces
better assessed candidates. CodeAct has not yet demonstrated a Builder or gameplay gain in
this solver. The current prototype is frozen with known protocol, authority, recovery, and
experiment-parity blockers; see
[`BUILDER_PROPOSAL_FREEZE_AUDIT.md`](../solvers/tycho_nooa/BUILDER_PROPOSAL_FREEZE_AUDIT.md).

## 1. What Builder is now

The current Builder is already a separate NOOA `Agent`, not a Tycho subprocess copied into
the solver:

1. `TriggerBuilderLoop` fires from a Tycho divergence/level-start signal or an Actor request.
2. It freezes an evidence frontier and begins a `TheoryTransaction` when the transactional
   treatment is active.
3. It constructs a fresh `FocusedBuilder` for that one fire. The Builder shares the LLM and
   workspace contract, but not the Actor's event history.
4. `BuilderStrategy` renders Tycho's pinned Builder prompt, attaches verifier output, and
   repeatedly calls the model with Tycho workspace tools.
5. Reads and `run_python` investigate evidence. Edits to `world_model.py` are intercepted by
   `WorkspaceToolSurface` and assessed by the host transaction.
6. Each assessed edit is compared with the current working incumbent over the complete
   frozen obligation table. A regression is restored; a named gain with no protected loss
   may become the working incumbent.
7. When the pass ends, the host finishes the transaction, decides publication, and gives the
   Actor an advisory report. The Builder never takes an ARC action.

```text
trigger / Actor request
          |
          v
freeze EvidenceFrontier + begin TheoryTransaction
          |
          v
fresh FocusedBuilder (fresh NOOA event history)
          |
          v
LLM -> one native tool call/batch -> result -> LLM -> ...
          |                         |
          |                         +-- read / run_python / notes
          +---------------------------- candidate edit
                                             |
                                             v
                           host evaluate -> retain or restore
                                             |
                                             v
                            finish transaction -> publish/withhold
                                             |
                                             v
                                  advisory Builder report -> Actor
```

### Ownership that must survive

| Thing | Owner | Builder access |
|---|---|---|
| Observed frames, transitions, terminal events | ARC host / workspace | Read-only evidence |
| Builder's temporary computations | One Builder invocation | Disposable |
| Candidate theory | Builder proposes | Noncanonical until host acceptance |
| Obligation definitions and evaluator | Host | Invoke, never rewrite |
| Canonical `world_model.py` | Host publication transaction | No direct publication authority |
| Environment action | Actor + host | Unavailable |
| Terminal truth | ARC host | Evidence only |

## 2. What CodeAct would actually change

The native Builder already has `run_python`, so the gain is not merely "it can write code."
The gain is a different unit of cognition.

Current:

```text
model call -> run_python string -> bounded stdout -> model call
model call -> run_python string -> bounded stdout -> model call
model call -> edit_function -> verifier text -> model call
```

CodeAct candidate:

```python
transitions = evidence.transitions(level=0)
cases = group_by_action(transitions)
candidate = theory.replace_function("transition", inferred_transition(cases))
receipt = await workbench.evaluate(candidate, claim="encode horizontal gap filling")

if receipt.retained:
    return_result(BuilderReport.from_receipt(receipt, open_questions=[...]))
```

The useful differences are:

- live local objects survive across cells within the build pass;
- loops, filtering, comparison, source transformation, and evaluation can be composed in
  one cell;
- only selected output enters model context instead of every intermediate array or grid;
- the final result can be a typed `BuilderReport`, eliminating prose parsing and the forced
  report fallback;
- NOOA records source cells, Python outputs, nested calls, and the typed return as events.

The current 16k transactional replay is a concrete example of the opportunity. Its Builder
made 14 completed/rejected workspace calls: one `ls`, three `read_file`, nine
`run_python`, and one notes write. The nine analysis calls repeatedly re-imported `wmlib`,
reloaded frames, and printed one more slice. It reached commitment mode without submitting a
semantic `world_model.py` edit, so the gate correctly failed. CodeAct may let the model keep
the frames, segments, hypotheses, and source transformation in one working namespace. That
is a plausible advantage, not yet an observed one.

## 3. The unsafe shortcut

Do **not** change this:

```python
@strategy(BuilderStrategy())
async def _construct(...): ...
```

to this:

```python
@strategy(CodeActStrategy(...))
async def _construct(...): ...
```

on the present `FocusedBuilder`.

`FocusedBuilder` holds `_executor`, `_tool_surface`, and other privileged objects. NOOA's
current sandbox replaces `self` with a broker proxy, but the broker resolves arbitrary
attribute paths, including legitimate private attributes. `hidden_field` controls generated
documentation; it is not an authorization boundary. A cell could therefore reach
`self._executor`, `self._tool_surface`, or nested mutation methods and bypass the intended
candidate-submission path. Kernel filesystem restrictions do not fix a broker that willingly
executes a privileged parent-side method.

This is the Builder version of the Actor lesson: a process boundary is not enough when the
RPC interface exports the authority you meant to contain.

## 4. Recommended shape

```text
                         HOST / CONTROL REALM

 ARC evidence ---> serialize frozen case -------------------------+
 canonical WM ---> copy incumbent ----------------------------+   |
 obligations ----> immutable descriptions -----------------+   |   |
                                                          |   |   |
                         CONSTRUCTION REALM                |   |   |
                  (fresh per Builder invocation)           |   |   |
                                                          |   |   |
        +-------------------------------------------+       |   |   |
        | FocusedCodeActBuilder                     |       |   |   |
        |                                           |       |   |   |
        | persistent REPL locals for this pass      |       |   |   |
        | realm-local evidence / copied incumbent <--+-------+---+---+
        | wmlib + symmetry + source-edit helpers    |       |
        | no ARC action, no canonical workspace     |       |
        +----------------------+--------------------+       |
                               |                            |
                     CandidateSubmission                   |
                     candidate bytes + claim                |
                               |                            |
                               v                            |
                    pure host evaluator in separate realm <--+
                               |
                 immutable EvaluationReceipt
                               |
                               +---- feedback to CodeAct session
                               |
                               v
                      typed BuilderReport
                               |
                               v
                TheoryTransaction.finish(report, receipts)
                               |
                               v
                      advisory report -> Actor
```

### Model-facing interface

The narrow interface should describe scientific work, not files. Most of it should be one
realm-local argument rather than a family of query tools:

```python
class FrozenBuildCase(BaseModel):
    evidence: FrozenEvidence
    incumbent: TheoryDraft
    obligations: tuple[ObligationDescription, ...]
    metadata: BuildMetadata

class CandidateSubmission(BaseModel):
    case_sha256: str
    declared_parent_source_sha256: str
    parent_bound_patch: tuple[ByteEdit, ...]
    candidate_source: BlobReference
    provenance: CandidateProvenance

class BuilderReport(BaseModel):
    disposition: Literal["candidate_submitted", "no_candidate", "inconclusive"]
    confidence: Literal["none", "low", "medium", "high"]
    retained_record_id: str | None
    dynamics: str
    outcome: str
    open_questions: list[str]
    discriminating_probe: str | None
    recommended_action: ActionDecision | None
    notes: str
```

`evaluate(submission)` is the only request channel in the CodeAct loop. It is a pure query
with respect to canonical and transaction state: it compiles and replays candidate bytes in
a separate evaluator realm and returns an immutable receipt. It does not retain, restore,
publish, or change a working incumbent. The host may durably append the request and receipt
to an audit ledger, but that record is evidence of the submitted request, not a Builder-owned
state transition.

The Builder can evaluate a candidate DAG by naming a parent and supplying a parent-bound
patch in every submission. A parent name alone cannot reveal stale-local-state construction;
the host must apply the patch to the exact named parent and reproduce the candidate bytes.
`TheoryTransaction.finish(report, receipts)` validates the complete receipt set, chooses a
Pareto-safe result, and performs the only retention/publication mutation.

Retained prior candidates are excluded from the first experiment. They are a later ablation,
not part of the v1 case.

### Do not pass host objects by reference

`FrozenBuildCase` should be live and convenient in the REPL, but it must be a realm-local
snapshot produced by schema-validated serialization, read-only mappings, immutable DTOs,
and non-writeable array storage. It must not contain a proxy, closure, file descriptor,
method, cache, lazy loader, or nested object that leads back to the host.

NOOA pass-by-reference is cognitively attractive, but it conflicts with causal poverty when
the referenced graph contains host authority. The property we want is not “the prompt calls
this object frozen.” It is “mutating or traversing this object can affect only disposable
realm-local state.” Content-addressed read-only blobs or memory maps are acceptable when the
realm has no writable host mapping.

### Construction realm options

The smallest safe treatment is a disposable workspace copy:

- mount/copy only the committed evidence frontier, incumbent theory, trusted helpers, and
  scratch directory;
- disable network and credentials;
- let CodeAct manipulate candidate source in memory or in the disposable copy;
- accept only schema-validated candidate bytes over a narrow request protocol; avoid pickle
  or arbitrary-object IPC;
- run candidate evaluation in a distinct host-controlled evaluator realm with a strict
  evaluation-count/CPU budget;
- copy nothing back except typed submissions, receipts, event traces, and the final report.

Before live integration, either NOOA's `self.*` broker needs a host-enforced capability
allowlist, or the CodeAct Builder agent must contain no route to canonical workspace,
evaluator, LLM, or action authority beyond the declared broker methods. Prompt hiding is not
sufficient.

## 5. What to keep, replace, and delete if it wins

### Keep

- `TriggerBuilderLoop` and Tycho's trigger/orchestrator distinction;
- one fresh Builder agent per fire;
- Builder deadlines and the Actor time reserve;
- `EvidenceFrontier`, trusted-runtime commitments, and immutable evidence;
- `TheoryTransaction`, complete obligation comparison, restore, and publication journal;
- Builder/Actor ownership enforcement and zero environment-action authority;
- run-wide LLM, decision-checkpoint, transaction, and uptake traces.

### Replace

- `BuilderStrategy._construction_loop` with NOOA's CodeAct loop;
- generic file editing with typed candidate submission;
- prose report parsing with a typed `BuilderReport` return;
- repeated stateless `run_python` calls with session-local computation over frozen evidence.

### Delete after parity is proven

- manual native tool-call validation/dispatch inside `BuilderStrategy`;
- forced final-report calls and rejected final-report tool-call bookkeeping;
- report freshness inferred from file hashes;
- forced `notes/world_model.md` breadcrumbs, if typed reports/notes become the single durable
  Builder handoff rather than another parallel truth.

Do not add typed Builder state beside `notes/world_model.md`, reports, retained candidates,
and transaction records indefinitely. A new canonical representation must make an old one
derived or unnecessary.

## 6. State and agent lifetime

Keep Actor and Builder separate. Keep the Builder fresh per invocation.

For Builder CodeAct:

```text
REPL locals                 one build pass; disposable scientific working state
BuilderReport/open questions durable handoff to the next pass
retained candidate records  executable attempts with exact lineage
canonical world_model.py     only host-published theory
```

There is little value in persistent `self` state on a fresh Builder. Cross-fire knowledge is
valuable only when it is inspectable and tied to evidence/candidate lineage. Put that in a
typed report or retained record, not an invisible long-lived object. If a later experiment
tests a persistent Builder, treat agent lifetime as its own intervention.

## 7. Smallest discriminating experiment

Do not integrate CodeAct into live ARC first. Run a Builder-only frozen replay.

### P0: zero-provider contract proof

Build golden paths for both interfaces over the same frozen Builder snapshot:

1. **Evidence composition:** recover the known gap-filling mechanics from several remote
   transitions.
2. **Semantic repair:** submit the known retained candidate and obtain the expected complete
   obligation receipt.
3. **Coupled repair:** present an obvious local fix that repairs one failed transition but
   breaks a remote protected obligation. Pass only if the Builder discovers a general repair
   that fixes the target without the remote regression.
4. **Correct abstention:** report `no_candidate` when the frozen evidence cannot distinguish
   two live models.
5. **Authority attack:** attempt self traversal, `object.__getattribute__`, closure/global
   recovery, imports, filesystem writes, symlinks, pickle/object IPC, subprocess, network,
   inherited file descriptors, fork/clone, monkey-patching, destructors/finalizers, exception
   payloads, nested NOOA calls, and mutation through every passed object.
6. **Recovery:** interrupt after submission/evaluation and verify canonical bytes and journal
   recovery.

No provider calls and no ARC actions. The gate is interface correctness, complete traces,
and failure-closed authority.

The governing behavioral invariant is:

> No generated Python program in the Builder realm can cause an externally persistent state
> transition except by emitting a schema-valid `CandidateSubmission` request or final
> `BuilderReport`.

A finite attack suite does not prove this universal statement. The construction must carry
most of the proof: separate process and identity, closed inherited descriptors, no authority
in the object graph, read-only mounts, disposable scratch, network denial, no fork/clone,
non-executable typed IPC, host rate limits, and canonical-state hash checks before and after
every session. The attack suite tests that construction and preserves counterexamples.

### P1: paid, zero-ARC matched replay

Use one or two preserved evidence frontiers where the current Builder's outcome is already
known. Hold constant:

- model, provider route, prompt's semantic assignment, frozen evidence, incumbent theory;
- trusted helpers, evaluator, obligation table, transaction/publication policy;
- wall-clock, token, and dollar caps;
- final `BuilderReport` schema and claim ceiling.

Do **not** match on tool-call count. Native calls and CodeAct cells carry different amounts
of computation. Match the scarce resources: provider tokens/cost, elapsed allocation, and
evidence available.

Primary outcome:

1. **total model plus evaluator cost to the first unique Pareto-safe semantic candidate**.

Important intermediate outcomes:

1. cost to first protected obligation gain, even if the candidate is not yet Pareto-safe;
2. publication disposition and named obligation gains/losses;
3. whether the final report is supported by submitted evidence and receipts.

Secondary outcomes:

- funnel timing and cost: evidence inspection -> first candidate -> first semantic candidate
  -> first obligation gain -> first Pareto-safe gain -> published improvement;
- candidate yield: unique Pareto-safe gain candidates / unique evaluated candidate hashes;
- cost-normalized yield: unique protected gains / total model plus evaluator cost;
- provider calls, tokens, cells/native calls, evidence queries, candidate attempts;
- context bytes and truncation;
- invalid tool use, typed failures, recovery correctness;
- trace reconstructability.

### Result-to-action map

| Result | Action |
|---|---|
| CodeAct reaches the first unique Pareto-safe semantic candidate at lower total model plus evaluator cost, with complete authority and trace gates | Implement behind a `builder_codeact` treatment; still no default switch |
| Same candidate quality but materially lower cost/time or simpler code | Consider migration only if the deletion is real and trace quality does not fall |
| No assessed candidate in either arm | Improve the candidate/evidence interface; do not infer model incapability |
| CodeAct wins only by seeing more evidence or using a broader evaluator | Invalid comparison; repair parity |
| Any canonical-write, evaluator, manifest, or private-`self` bypass | Fail closed; fix authority before another provider run |
| Native equals or beats CodeAct | Keep native Builder; CodeAct remains an optional model-specific strategy |

## 8. Implementation order

1. Specify the four value contracts and canonical hashing.
2. Implement realm-local snapshot construction.
3. Implement the pure evaluator and immutable receipt ledger.
4. Implement `TheoryTransaction.finish()` independently of Builder execution mode.
5. Pass structural authority and semantic noninterference tests.
6. Pass friendly repair, coupled repair, unsupported-generalization, stale-lineage,
   determinism, and interruption/recovery tests.
7. Implement Native and CodeAct adapters over the identical protocol.
8. Run a zero-ARC matched replay.
9. Decide whether CodeAct has earned live Builder integration.

## 9. Claim boundaries

### Observed

- Builder is fresh per fire and already runs on NOOA's `Agent`/event/runtime chassis.
- Its inner construction loop is custom native-tool orchestration, not NOOA CodeAct.
- The transactional path freezes evidence, assesses edits, restores rejected candidates, and
  lets the host decide publication.
- A preserved 16k replay spent most of its calls on repeated evidence analysis and submitted
  no semantic model edit.
- NOOA CodeAct provides persistent per-call locals, typed return validation, evented Python
  cells, and a sandbox broker for `self.*`.
- That broker permits private attribute paths; `hidden` affects visibility/documentation, not
  parent-side authority.

### Inference

- Persistent local composition may reduce Builder analysis fragmentation and get to an
  assessed candidate sooner.
- Typed submission/report contracts can replace several report/file orchestration patches.
- A generic live `self` on the present Builder would weaken the transaction boundary.

### Unknown

- Whether the target models use CodeAct better than the current native Builder interface.
- Whether candidate quality, cost, or gameplay improves.
- Whether the v1 immutable case and pure evaluator are sufficient for productive candidate
  construction without retained prior candidates.
- Whether retained prior candidates add value in a later matched ablation.

## Source map

- Current Builder loop and fresh agent: `solvers/tycho_nooa/src/tycho_nooa/builder.py`
- Trigger, deadline, transaction finish, and fresh construction: `solvers/tycho_nooa/src/tycho_nooa/trigger_builder.py`
- Tool ownership and edit interception: `solvers/tycho_nooa/src/tycho_nooa/tools.py`
- Transactional evaluator wiring: `solvers/tycho_nooa/src/tycho_nooa/agent.py`
- Monotone transaction and recovery: `solvers/tycho_nooa/src/tycho_nooa/theory_transaction.py`
- Complete obligation comparison: `solvers/tycho_nooa/src/tycho_nooa/theory_evaluation.py`
- Pinned Builder prompt: `solvers/tycho_nooa/.venv/lib/python3.12/site-packages/tycho/prompts/builder.system.j2`
- NOOA CodeAct loop: `/home/hipra/projects/nooa/src/nooa/strategies/codeact.py`
- NOOA sandbox broker: `/home/hipra/projects/nooa/src/nooa/runtime/sandbox/executor.py` and `worker.py`
- Preserved 16k Builder gate: `analysis/theory-transaction/20260822T182634Z-deepseek-16k-gate-ar25/`
