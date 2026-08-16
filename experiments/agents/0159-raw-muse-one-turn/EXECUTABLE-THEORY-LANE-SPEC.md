# E0 / WM0 / WM1 / WM2 / P0 Executable Theory Lane Interface Specification

Status: accepted design; E0, WM0, and the WM1 quick-theory refinement are
implemented; P0 is not yet present. Muse and GPT-5.6 Luna WM0 first contacts are
preserved; neither authored a candidate under the original full-contract-only
interface.

The first WM1 contact was also valid but authored no candidate. It produced two
successful broker read events whose bare-expression CodeAct cells had empty
stdout. This identifies return-value visibility as a prerequisite interface
fix before another WM1 evidentiary run; it does not falsify the quick-theory
contract.

The recovery-feedback check then exercised automatic display successfully on a
bare broker call, but Muse never called the world-model broker and authored no
candidate. Thus result visibility is proven locally and in-run, while theory
discoverability versus low immediate utility remains unresolved.

GPT-5.6 Luna then used the same WM1 surface for eight valid actions and likewise
made no world-model call. It formed a more economical implicit navigation model
with ten analysis cells. Cross-model non-adoption weakens a Muse-specific
explanation while leaving discoverability versus utility unresolved.

A subsequent one-turn Luna capability probe correctly reconstructed the lane
and broader harness, then explicitly deferred theory creation before observing
a transition. WM2 therefore adds modular practice guidance and a transparent
Theory Workbench View rather than another unexplained availability reminder.
Its requirements are frozen separately in
[`WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md`](WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md).

This specification makes
[`docs/adr/0003-executable-theory-lane-for-0159.md`](../../../docs/adr/0003-executable-theory-lane-for-0159.md)
executable as an engineering contract. Normative terms **MUST**, **MUST NOT**,
**SHOULD**, and **MAY** have their usual requirements meaning.

## 1. Scope

The lane extends the persistent 0159 NOOA investigator with:

- **E0**: a typed host-evidence adapter;
- **WM0**: brokered model-authored executable commitments with isolated replay;
- **WM1**: a low-threshold partial-predictor contract beside the full Moore
  machine, with model-authored promotion between them;
- **WM2**: separately ablatable practice guidance and a deterministic theory
  opportunity surface over E0 evidence and lane state;
- **P0**: automatic pre-action shadow prediction and post-successor scoring.

### Decision-local observation input

Every CodeAct variant receives one model-facing view of the current public
observation: `image`, a 64x64 palette-mode PIL image. `show(image)` supplies the
visual view; `image.size`, `image.crop(...)`, and `np.asarray(image)[y, x]`
provide ordinary exact computation over the same object.

The worker validates the private row-major cell transport as non-empty,
rectangular, integer-valued, and within the ARC palette before constructing the
image. It contains no segmentation, object identity, role, inferred mechanic,
or goal. Mutating the decision-local PIL object cannot alter host evidence.
NOOA's media wrapper remains transport plumbing and is not model-visible in
CodeAct variants.

The durable artifact is an External Executable Reconstruction containing
provisional Executable Conjectures. It may serve as a Working Theory, but no
interface or status calls it correct or complete.

This specification does not add:

- planning;
- a world-model builder;
- a second actor;
- shell access;
- action batching;
- automatic belief revision;
- host-authored game semantics;
- a causal claim that executable content influenced action choice.

### 1.1 WM1 contract ladder

WM1 accepts either of two complete candidate contracts through the same atomic
broker operation:

1. A **partial predictor** defines `predict(grid, action) -> grid | None`. The
   returned grid MUST match the observation shape, contain integers from `-1`
   through `15`, and uses `-1` as cell-level abstention. `None` abstains for the
   whole action. Each ordinary transition is evaluated independently; attempt
   boundaries are never replayed as dynamics.
2. A **Moore machine** retains WM0's `State`, `init_state`, `transition`,
   `render`, and `outcome` contract and stateful attempt-segmented replay.

Operational validity controls activation for both contracts. The actor promotes
a quick theory by authoring a later full Moore source; the broker and evaluator
MUST NOT synthesize, rewrite, or promote it. Prose remains first-class for
competing explanations that are not ready for executable commitment.

### 1.2 WM2 taught interface

WM2 does not add another source contract or change activation. It teaches
game-general practices for the existing WM1 contract and continuously renders
mechanical evidence/artifact conditions. The host may identify an exact
Opportunity Condition; only the model may judge that it constitutes a useful
opportunity. The two WM2 modules remain independently configurable and are
specified in `WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md`.

## 2. Ownership and authority

| Axis | Authority |
|---|---|
| Public observation and action evidence | Python 3.14 ARC host |
| Evidence classification | Deterministic host adapter |
| Game interpretation and source semantics | Muse |
| Prose hypothesis authorship | Muse |
| Executable commitment authorship | Muse |
| Mutation mediation | Python 3.12 worker broker |
| Canonical source storage | Run-local host filesystem |
| Candidate execution | Fresh isolated evaluator |
| Evaluation semantics | Pinned deterministic verifier |
| Environment action legality and execution | Python 3.14 ARC host |
| Prediction scoring | Deterministic host/evaluator machinery |
| Replacement of the current commitment | Muse through a new valid edit |

Evaluation **MUST NOT** modify source. Measurement **MUST NOT** alter the current
decision or issue an environment action.

## 3. Components

### 3.1 Host evidence ledger

The host ledger is append-only and authoritative. Every committed record receives
a monotonically increasing `host_sequence` and stable `event_id`.

The ledger owns:

- observations and settled grids;
- accepted decisions;
- environment-step invocation;
- action results;
- public `state`, `levels_completed`, and `full_reset`;
- animation side-channel references;
- shadow receipts and prediction scores;
- integrity and failure records.

### 3.2 Executable theory broker

The broker lives in the persistent worker but exposes no worker internals to
candidate code. It owns:

- model-facing read and edit affordances;
- source validation before storage;
- monotonic version allocation;
- immutable evaluation-bundle assembly;
- invocation of the isolated evaluator;
- atomic current-version replacement;
- bounded feedback rendering.

### 3.3 Isolated evaluator

Each invocation uses a fresh process or container. It receives one immutable
bundle and returns one structured result. It has no mutation authority over the
canonical workspace.

### 3.4 Shadow coordinator

The host coordinates pre-action prediction and later scoring. The shadow result
is not returned to Muse before the already accepted action is executed.

## 4. Identifiers and commitments

The implementation **MUST** define stable, run-scoped identifiers for:

- `event_id`;
- `evidence_snapshot_id`;
- `attempt_id`;
- `candidate_id`;
- `model_version_id`;
- `evaluation_id`;
- `prediction_receipt_id`;
- `prediction_score_id`.

Identifiers **MUST NOT** derive authority from timestamps. Content commitments
use full SHA-256 digests over a documented canonical encoding. Timestamps are
diagnostic metadata only.

Every record that references a blob **MUST** include its content digest. Blobs
required to reproduce a score **MUST** be preserved, not represented only by a
digest.

## 5. E0: typed evidence adapter

### 5.1 Evidence snapshot

An `EvidenceSnapshot` is an immutable, prefix-bound view of authoritative host
evidence:

```python
@dataclass(frozen=True)
class EvidenceSnapshot:
    snapshot_id: str
    through_host_sequence: int
    occurrence_prefix_sha256: str
    transition_records_sha256: str
    source_event_ids: tuple[str, ...]
    occurrence_records_json: str
    transitions: tuple[EvaluationTransition, ...]
    terminal_events: tuple[TerminalEvent, ...]
    animation_events: tuple[AnimationEventRef, ...]
```

`occurrence_records_json` is the canonical JSON encoding of the complete
authoritative prefix. It makes the input commitment independently
recomputable from the snapshot rather than trusting a digest supplied beside
unavailable input. `source_event_ids` makes append-only prefix comparison
mechanical. Both fields are model-hidden evaluation infrastructure.

The snapshot **MUST** bind the exact authoritative input prefix and the complete
derived evaluation manifest. A matching input hash alone does not establish
that classification was correct.

### 5.2 Transition schema

```python
@dataclass(frozen=True)
class EvaluationTransition:
    source_event_id: str
    global_turn: int
    action: Action
    before_observation_id: str
    after_observation_id: str
    before_grid_sha256: str
    after_grid_sha256: str
    public_state_after: str
    levels_completed_before: int
    levels_completed_after: int
    full_reset_after: bool
    boundary: Literal["none", "explicit_reset", "api_full_reset"]
    outcome: Literal["ongoing", "level_complete", "game_over"]
    attempt_before: str
    attempt_after: str | None
    animation_event_id: str | None
```

`boundary`, `outcome`, and animation are independent channels. The adapter
**MUST NOT** force them into one mutually exclusive classification.

`attempt_before` and `attempt_after` are evaluator-local replay-segment
identifiers. They are not added to canonical model-visible observation
occurrences and do not assert a hidden game ontology.

### 5.3 Attempt allocation

The adapter **MUST** apply these rules in order:

1. The initial host `environment.reset()` allocates the first attempt.
2. An accepted explicit `RESET` decision closes `attempt_before` and allocates
   exactly one `attempt_after` for its successor.
3. API `full_reset=True` closes `attempt_before` and allocates exactly one
   `attempt_after`, unless rule 2 already allocated it for the same transition.
4. API `GAME_OVER` closes `attempt_before`; `attempt_after` is `None` until the
   host actually performs a reset.
5. A `levels_completed` increase closes `attempt_before`; its terminal successor
   remains terminal evidence for that attempt. If play continues, the next
   public decision observation allocates the next attempt.
6. A visually reset-like transition with action other than explicit `RESET`,
   `full_reset=False`, unchanged `levels_completed`, and nonterminal public state
   remains in the same attempt.
7. Ordinary transitions have `attempt_before == attempt_after`.

The adapter **MUST NOT** stitch replay state across unequal attempt identifiers.

### 5.4 Outcome allocation

- `level_complete` requires `levels_completed_after > levels_completed_before`.
- `game_over` requires the public API state classified as terminal game over.
- Reset-like `NOT_FINISHED` transitions remain `ongoing`.
- No pixel pattern may independently assign an outcome.

### 5.5 Animation

The settled final grid remains the action-result observation. A multi-frame
public response also creates an animation reference whose manifest binds every
frame, order, settled-frame index, and blob digest. Animation frames **MUST NOT**
be silently inserted as ordinary decision transitions.

### 5.6 Snapshot monotonicity

For one run, `ExecutableTheoryLane.advance(snapshot)`:

- **MUST** accept the current snapshot idempotently;
- **MUST** accept a strict append-only successor;
- **MUST** reject an older frontier;
- **MUST** reject a divergent snapshot at an already installed frontier;
- **MUST** install the snapshot atomically;
- **MUST** score every newly resolvable pending shadow exactly once;
- **MUST** report the identifiers and statuses of resolved shadows in its
  returned `LaneState`.

```python
@dataclass(frozen=True)
class LaneState:
    evidence_snapshot_id: str
    current_model_version_id: str | None
    resolved_prediction_scores: tuple[str, ...]
    still_pending_prediction_receipts: tuple[str, ...]
```

## 6. WM0: executable commitment lifecycle

### 6.1 Executable contract

The initial 0159 contract is deterministic and uses the human-facing action
vocabulary:

```python
@dataclass
class State:
    ...

def init_state(grid0: np.ndarray, level: int) -> State:
    ...

def transition(state: State, action: dict) -> State:
    # {"action": "UP", "x": None, "y": None}
    ...

def render(state: State) -> np.ndarray:
    ...

def outcome(state: State) -> str:
    # "ongoing" | "level_complete" | "game_over"
    ...
```

`actions`, `subgoals`, `heuristic`, `planner_key`, custom planner files, and a
builder are outside WM0.

### 6.2 Model-facing affordances

```python
self.read_world_model()
self.write_world_model(source)
self.edit_world_model_function(name, source)
```

The methods are brokered. Candidate code never receives the persistent worker's
`self` object.

CodeAct executes scripts rather than an interactive REPL. To keep this detail
from swallowing tool feedback, a top-level final bare call to any allowed
broker method **MUST** be rewritten before sandbox execution to display its
return value. The original and effective cell source, transformation flag, and
result remain audited. Assignment and explicit printing retain ordinary Python
semantics and are not rewritten.

`read_world_model()` returns:

- current source;
- current version id and source digest;
- installed evidence snapshot id;
- latest structured evaluation summary;
- whether an unresolved shadow exists.

It **MUST NOT** inject repair advice.

Every broker result includes a `tool_feedback` envelope with:

- `status`: `success`, `info`, `warning`, or `error`;
- `operation` and lifecycle `phase`;
- stable mechanical failure `code`, when applicable;
- a plain-language `message`;
- the exact `state_effect`, especially whether the previous model was preserved
  or the candidate became current;
- bounded diagnostics and first-divergence evidence;
- a `retry` object naming whether another model-authored call is allowed, the
  relevant version identities, and the available broker calls.

The envelope describes recovery mechanics, not game semantics. It **MUST NOT**
author source, select a hypothesis, recommend an action, or silently retry.

### 6.3 System-facing interface

```python
class ExecutableTheoryLane:
    def advance(self, snapshot: EvidenceSnapshot) -> LaneState:
        """Atomically install evidence and score newly resolved predictions."""

    def apply(
        self,
        edit: ModelEdit,
        provenance: EditProvenance,
    ) -> ModelEvaluation:
        """Store, isolate, execute, evaluate, and conditionally activate one edit."""

    def shadow(self, decision: Decision) -> PredictionReceipt:
        """Create and durably commit one pre-action prediction."""
```

Model-facing whole-file and function edits are translated internally into one
`ModelEdit`. Callers do not separately save, verify, version, or promote.

### 6.4 Apply transaction

`apply()` performs one transaction:

1. Bind the edit to the currently installed evidence snapshot and author event.
2. Allocate a candidate id.
3. Construct candidate source without mutating the current source.
4. Persist candidate source and provenance.
5. Assemble an immutable evaluation bundle.
6. Execute operational checks and empirical replay in isolation, producing one
   complete structured result.
7. If operationally invalid, persist the rejection and retain the old current
   version.
8. If operationally valid, classify the already completed empirical result,
   allocate a model version, and atomically persist the version, evaluation, and
   new current-version pointer.
9. Return bounded structured feedback to Muse.

A process crash before the current-version pointer swap **MUST** leave the old
version current. The version, complete evaluation, and pointer swap are one
atomic commit; a crash after it **MUST** be recoverable from that committed
record. There is no state in which current source changed but version, evidence
frontier, or evaluation identity is unknown.

An evaluator infrastructure failure is not evidence that candidate source is
operationally invalid. It preserves the candidate as an unevaluated provenance
artifact, leaves the old version current, and returns a typed lane failure rather
than a `ModelEvaluation`. Retrying that candidate requires an explicit frozen
retry rule; the broker may not silently rerun it until one result appears.

### 6.5 Operational validity

A candidate is `rejected_operationally` when any required operational check
fails, including:

- parse or syntax failure;
- prohibited import or operation;
- missing required callable;
- invalid required function contract discovered by bounded contract smoke;
- process termination;
- memory, CPU, wall-time, process, file-size, or output-limit breach before a
  bounded evaluation result can be produced;
- attempt to access authority outside the immutable evaluation bundle.

The rejection is preserved with candidate source digest, author provenance,
evidence frontier, failure code, and bounded diagnostics. It does not receive a
model version id and does not replace the current version.

### 6.6 Empirical status and activation

An operationally valid candidate becomes current regardless of empirical score:

```python
CandidateStatus = Literal[
    "rejected_operationally",
    "active_contradicted",
    "active_inconclusive",
    "active_supported",
]
```

- `active_contradicted`: at least one applicable deterministic empirical check
  contradicts the candidate, including a bounded per-evidence execution error.
- `active_inconclusive`: no applicable contradiction was found, but evidence is
  insufficient or coverage is vacuous for at least one required claim channel.
- `active_supported`: at least one non-vacuous applicable check exists for every
  claimed required channel and all such checks support the candidate so far.

The result **MUST** report transition and outcome channels separately. A
supported transition channel with no terminal evidence does not produce a
supported outcome channel.

There is no `correct`, `true`, `verified`, or automatically promoted status.

### 6.7 Structured evaluation

```python
@dataclass(frozen=True)
class ModelEvaluation:
    evaluation_id: str
    candidate_id: str
    candidate_status: CandidateStatus
    active_model_version_id: str | None
    previous_model_version_id: str | None
    source_sha256: str
    evidence_snapshot_id: str
    operational: OperationalResult
    initial_render: RenderEvaluation | None
    dynamics: DynamicsEvaluation | None
    outcome: OutcomeEvaluation | None
    first_divergence: Divergence | None
```

Required quantitative fields include, when applicable:

- exact and known-cell accuracy;
- prediction coverage;
- evaluated, matched, changing, and no-op counts;
- per-attempt and per-level partitions;
- observation-variant usage;
- first divergent source transition;
- observed and predicted delta blob references;
- terminal false positives and false negatives;
- bounded execution errors.

### 6.8 Feedback ceiling

Model-visible feedback may state:

- operational failure code and location;
- initial render mismatch;
- transition and cell scores;
- prediction coverage;
- first divergent transition and action;
- compact observed and predicted deltas;
- outcome evidence counts and misclassification type.
- whether the operation mutated canonical model state;
- whether another explicit model-authored edit is allowed and which broker
  calls are available.

It **MUST NOT**:

- edit source;
- name a repair function as an instruction;
- say "fix" or provide a repair focus;
- infer or name object roles;
- choose hidden state;
- recommend an environment action;
- select among Muse's hypotheses.

### 6.9 Version record

Every operationally valid version binds:

```json
{
  "model_version_id": "...",
  "candidate_id": "...",
  "author_event_id": "...",
  "evidence_snapshot_id": "...",
  "source_sha256": "...",
  "previous_model_version_id": "...",
  "evaluation_id": "...",
  "candidate_status": "active_contradicted"
}
```

Rejected candidates receive equivalent provenance without a model version or
current-pointer mutation.

## 7. Containment contract

### 7.1 Evaluation bundle

The broker constructs a fresh bundle containing only:

- candidate source;
- pinned evaluator and support-library source;
- the immutable evidence projection required for this evaluation;
- a manifest of every file and digest;
- a generated entry point with no secrets.

The canonical workspace is not mounted. Provider keys, worker environment,
conversation state, history objects, IPC descriptors, and host source paths are
absent.

### 7.2 Runtime restrictions

The evaluator **MUST** enforce:

- no network;
- read-only root filesystem;
- read-only evaluation bundle;
- writable ephemeral temporary storage only;
- dropped capabilities and no privilege escalation;
- bounded processes and file descriptors;
- CPU, memory, wall-time, file-size, and output limits;
- sanitized environment variables;
- fresh process/container per invocation;
- captured and bounded stdout/stderr;
- fail-closed behavior when required isolation is unavailable.

Host execution mode is development-only and invalid for admissible live runs.

### 7.3 Containment acceptance probes

Before live execution, frozen tests **MUST** submit candidates that attempt to:

1. read a provider credential;
2. enumerate or read worker process state;
3. access inherited IPC file descriptors;
4. read outside the bundle;
5. overwrite candidate source, evidence, evaluator code, or version records;
6. create a network connection;
7. spawn or fork processes;
8. exhaust CPU;
9. exhaust memory;
10. flood stdout/stderr;
11. create oversized files;
12. use prohibited dynamic loading or reflection;
13. escape through symlinks or path traversal;
14. hang only on one observed transition after passing import.

Each probe **MUST** produce a deterministic operational rejection or bounded
empirical execution error according to the frozen classification rules. None may
alter canonical artifacts or suppress a subsequent host action.

## 8. P0: shadow prediction lifecycle

### 8.1 Authoritative ordering

For one environment turn, the host ledger order is normative:

```text
N     decision accepted
N+1   prediction receipt appended, flushed, and fsynced
N+2   environment-step invocation appended, flushed, and fsynced
N+3   successor observation/action result appended, flushed, and fsynced
N+4   prediction score appended, flushed, and fsynced
```

The receipt includes the decision event id. The score includes the prediction
receipt id and action-result event id. Timestamps do not establish ordering.

If no model exists or prediction cannot be produced, `N+1` is still a receipt
with an explicit unavailable status. The host then proceeds to `N+2`.

### 8.2 Prediction receipt

`shadow(decision)` predicts but does not score:

```python
@dataclass(frozen=True)
class PredictionReceipt:
    prediction_receipt_id: str
    decision_event_id: str
    status: Literal[
        "predicted",
        "no_executable_model",
        "model_operational_failure",
        "measurement_failure",
    ]
    model_version_id: str | None
    model_source_sha256: str | None
    evidence_snapshot_id: str
    attempt_id: str
    level: int
    global_turn: int
    state_anchor_grid_sha256: str
    action: Action
    canonical_grid_blob_sha256: str | None
    claimed_cell_mask_blob_sha256: str | None
    prediction_coverage: float | None
    observation_variant_blob_sha256s: tuple[str, ...]
    predicted_outcome: str | None
```

The canonical grid, claimed mask, and variants are stored as content-addressed
blobs. Variant ordering and acceptance semantics are frozen by the evaluator
contract.

### 8.3 Pending state

A `predicted` receipt becomes pending after durable append. At most one pending
receipt may refer to one accepted decision event. Repeating `shadow()` for the
same decision returns the existing receipt and does not execute the model again.

Unavailable receipts are terminal measurement records and are not pending.

### 8.4 Resolution

`advance(successor_snapshot)` resolves a pending prediction only when the
snapshot contains the exact action-result event linked to its decision.

The host invokes this advance after durably recording the successor and before
applying any terminal or action-cap stop. Scoring therefore does not depend on a
later model turn and still occurs when the successor completes a level or ends
the game.

Resolution writes:

```python
@dataclass(frozen=True)
class PredictionScore:
    prediction_score_id: str
    prediction_receipt_id: str
    action_result_event_id: str
    successor_evidence_snapshot_id: str
    status: Literal["scored", "unresolvable", "scoring_failure"]
    canonical_exact: bool | None
    accepted_variant_index: int | None
    known_cell_accuracy: float | None
    prediction_coverage: float | None
    strict_cell_accuracy: float | None
    outcome_match: bool | None
    observed_grid_sha256: str | None
```

Resolution is idempotent. A receipt receives at most one score. A divergent
successor reference is an integrity failure, not a new score.

If `environment.step()` fails or returns no authoritative successor, the
receipt remains unresolved and is linked to the host failure record. The system
does not fabricate an observation or retry the environment action.

### 8.5 Noninterference

Shadow prediction executes only after the decision is accepted. Its output is
not added to the active NOOA conversation before the corresponding environment
action. Failure to predict or persist rich blobs:

- **MUST NOT** change the decision;
- **MUST NOT** request another model decision;
- **MUST NOT** suppress the action;
- **MUST NOT** substitute a host action;
- **MUST NOT** execute the action twice.

Failure to append and fsync the minimal receipt before the environment-step
invocation is a P0 instrumentation failure. The experiment card must state
whether this invalidates only the prospective-prediction channel or the complete
run; it may not be decided after observing gameplay.

## 9. Interaction with notes and compaction

Existing durable prose notes remain first-class and retain their existing
model-facing interface in WM0. They are not automatically translated into code,
rewritten by the verifier, or moved solely for directory unification.

Context compaction:

- **MUST NOT** rewrite executable source or version records;
- **MUST** preserve the current model version id, source digest, evaluation id,
  and pending prediction ids in the recovery orientation;
- **MUST** treat canonical files and the evidence ledger as sources, not copy
  their full contents into the Summary;
- **MUST** resume with the same current-version pointer;
- **MUST** include model and prediction artifacts in frozen provenance manifests.

## 10. Failure taxonomy

At minimum, model-invisible audit and final reports distinguish:

- evidence adapter failure;
- evidence integrity failure;
- broker transaction failure;
- candidate operational rejection;
- evaluator infrastructure failure;
- empirical contradiction;
- shadow unavailable because no model exists;
- shadow model failure;
- shadow instrumentation failure;
- prediction scoring failure;
- provider/model decision failure;
- environment-action uncertainty;
- context compaction failure;
- gameplay noncompletion.

An empirical contradiction is not an infrastructure failure. A rejected
candidate is not automatically a failed game turn. A gameplay failure does not
prove the lane failed, and lane uptake does not prove benefit.

Every consequential infrastructure or authority-boundary failure preserves a
causal packet containing the relevant evidence prefix, source and configuration
hashes, model-visible interface, candidate or receipt, failure classification,
and smallest discriminating reproduction.

## 11. Source reuse and conformance

Tycho source pin:

```text
f68912a764372ead0a610db2e1c011d41ce5197e
```

Before live execution, `PROVENANCE.md` identifies:

- verbatim copied files/functions and their hashes;
- locally adapted files/functions;
- the reason for every adaptation;
- license and attribution;
- parity and conformance tests.

Tests are divided into:

1. **Source parity** for unchanged kernels.
2. **Structured-result parity** for equivalent Tycho/0159 fixtures.
3. **Adapter conformance** for action vocabulary, click coordinates, evidence
   boundaries, outcome channels, and animations.
4. **Feedback-policy tests** proving that retained structured measurements do
   not emit removed repair or action advice.

No document may simultaneously claim verbatim parity and materially altered
output for the same surface.

## 12. Acceptance gates before live contact

### E0 gate

- all event classifications pass frozen fixtures;
- reset boundaries cannot stitch attempts;
- visually reset-like ordinary transitions remain ordinary;
- animation and terminal evidence remain separate;
- snapshots are monotonic, idempotent, and divergence-rejecting;
- input and derived-manifest commitments verify independently.

### WM0 gate

- whole-file and function edits use one atomic apply transaction;
- operational rejection leaves the prior version current;
- operationally valid contradicted source becomes current;
- only Muse-authored valid edits replace current source;
- structured replay separates dynamics and outcome;
- feedback contains no forbidden semantic or procedural advice;
- all containment probes pass;
- scripted CodeAct traces can read, edit, receive feedback, and still return
  exactly one legal decision.

### P0 gate

- every accepted decision produces exactly one minimal receipt before step;
- receipts are flushed and fsynced before the step-invocation record;
- each successor resolves at most one receipt;
- no-model and prediction-failure paths still execute the accepted action once;
- shadow results cannot enter the current decision context;
- prediction blobs and scores are independently reproducible;
- step failure leaves the receipt unresolved without action retry.

Passing these gates licenses bounded first contact. It does not establish
gameplay benefit.

## 13. Experimental requirements

### 13.1 First-contact question

The cheaper first contact asks:

> Can Muse adopt the Executable Theory Lane and produce valid prospective
> predictions without breaking the existing persistent actor loop?

It is a treatment-only feasibility run. It reports:

- Reconstruction Activation or its absence;
- artifact creation and revision;
- retrospective replay by version and evidence frontier;
- automatic prospective prediction coverage and accuracy;
- dynamics and outcome channels separately;
- latency, provider calls, sandbox failures, tool exhaustion, and context growth;
- `levels_completed` and action count.

It cannot claim that the lane changed behavior, caused an action, or improved
gameplay.

### 13.2 Later matched contrast

If first contact passes containment and uptake criteria, freeze a prospective
control/treatment experiment:

```text
Control:   f3d8b3027508167a41a3de4ba5b125fec130575e capability surface
Treatment: cumulative Executable Theory Lane capability surface
```

Hold constant:

- core model;
- qualified provider policy;
- LS20 seed and fresh reset;
- action cap;
- provider-call cap;
- context window and compaction policy;
- human-facing action vocabulary;
- commitment interface;
- wall-time and provider-timeout policy;
- host evidence and audit policy except for treatment-required lane artifacts.

Historical development runs are not a matched control. Assignment, exclusions,
validity rules, outcome profile, and analysis must be frozen before protected
results are inspected.

### 13.3 Claim levels

Reports distinguish:

1. **Artifact ownership**: Muse authored or revised source.
2. **Reconstruction Activation**: the enabled actor created and prospectively
   used an external executable reconstruction. Source creation or retrospective
   fitting alone remains Model Theater. Automatic model-invisible P0 shadow
   prediction does not by itself establish actor use or activation.
3. **Retrospective validity**: a version replayed prior evidence.
4. **Prospective validity**: a frozen pre-action prediction matched its later
   successor.
5. **Consultation**: the action-producing trace read, edited, or evaluated the
   artifact.
6. **Causal-use candidate**: prospective content appears in a decision chain and
   nominates an intervention point.
7. **Local Reconstruction-Content Dependence**: a later matched intact-versus-
   perturbed shadow intervention changes the immediate response signature.
8. **Beneficial Causal Use**: the counterfactually changed decision improves an
   appropriate frozen external outcome.

WM0 first contact can reach levels 1 through 5 and nominate level 6. It cannot
establish levels 7 or 8.

## 14. Required implementation artifacts

Before the first live run, the cumulative experiment contains:

- an experiment card;
- `PROVENANCE.md`;
- frozen E0 schemas and fixtures;
- frozen WM0 source contract and candidate-status rules;
- frozen P0 receipt and score schemas;
- containment probe results;
- source and configuration commitments;
- scripted end-to-end traces for success and every consequential failure path;
- a run manifest naming the baseline commit and Tycho pin;
- an explicit claim ceiling.

Implementation begins only after this specification and the ADR agree. Changes
to event taxonomy, candidate activation, verifier feedback, shadow ordering,
noninterference, or the first-contact question require an explicit documentation
revision before another evidentiary run.
