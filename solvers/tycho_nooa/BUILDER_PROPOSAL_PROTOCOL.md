# Builder Pure Proposal Protocol

**Protocol:** Tycho Builder Proposal Protocol v1

**Status:** normative design contract; the current Python prototype is a
nonconformant frozen experiment documented in
[`BUILDER_PROPOSAL_FREEZE_AUDIT.md`](BUILDER_PROPOSAL_FREEZE_AUDIT.md)

**Scope:** Builder sessions that propose executable-theory changes

> [!WARNING]
> The current prototype does not satisfy this protocol's authority, recovery,
> exact-wire, or matched-replay obligations. It must not be used for live
> integration, hostile generated code, or a paid replay until the freeze audit's
> blockers are resolved.

## 1. Governing claim

A Builder session may transform a frozen case into proposals and reports, but
cannot directly mutate theory, transaction state, environment state, or host
authority.

The protocol separates four responsibilities:

~~~text
Builder computes
Host judges
Transaction remembers
Actor acts
~~~

The Builder realm is therefore a **pure proposal machine**. Persistent Python
locals, scratch files, temporary object mutation, exceptions, imports, and
execution order are internal implementation details. They have no direct
semantic effect outside the realm.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 2. Claim boundary

Version 1 specifies:

1. the four wire values exchanged around a Builder session;
2. their canonical representation and hashes;
3. legal state transitions;
4. candidate lineage and evaluation determinism;
5. the sole publication authority;
6. a semantic noninterference property; and
7. proof obligations for an implementation.

Version 1 deliberately excludes:

- retained prior candidates;
- direct Builder access to transaction or environment objects;
- arbitrary Python-object serialization;
- stochastic evaluators;
- live ARC integration; and
- claims that simpler or broader theories are better without an explicit,
  evidence-backed obligation that makes them so.

## 3. Authorities and realms

### 3.1 Host

The host constructs the frozen case, validates all messages, canonicalizes
candidate source, invokes the evaluator, records receipts, and calls
TheoryTransaction.finish().

The host is the only process that holds references or capabilities for:

- the canonical theory store;
- transaction working state;
- the environment or Actor;
- evaluator configuration;
- publication and restoration; and
- durable audit and accounting storage.

### 3.2 Builder realm

The Builder realm receives a realm-local FrozenBuildCase and may emit only:

1. CandidateSubmission messages during the session; and
2. at most one final BuilderReport.

The realm MUST NOT receive a live host object, callback, descriptor, closure,
file descriptor, lazy proxy, shared mutable buffer, or other capability that
can reach host authority.

### 3.3 Evaluator realm

The evaluator receives only the committed case, a host-reconstructed candidate,
the committed obligation universe, and committed evaluator inputs. It returns
an EvaluationReceipt. It has read-only inputs and no publication authority.

### 3.4 Theory transaction

TheoryTransaction is host-owned durable state. Its finish operation is the only
operation that may interpret a receipt ledger as a retention or publication
decision.

### 3.5 Actor

The Actor alone may cause environment transitions. No Builder or evaluator
message is an environment action.

## 4. Canonical representation

### 4.1 Canonical JSON profile

All protocol values use this restricted JSON profile:

- UTF-8 encoding without a byte-order mark;
- objects with unique string keys;
- keys sorted lexicographically by Unicode code point;
- arrays whose order is semantically significant;
- no insignificant whitespace;
- JSON booleans and null;
- base-10 integers only;
- no binary floating-point values, NaN, or infinities; and
- strings normalized to Unicode NFC.

Costs and other non-integral measurements MUST be encoded as canonical decimal
strings. Unknown fields MUST be rejected. Candidate source bytes are blobs, not
JSON strings, and are exempt from Unicode normalization.

The reference serialization is:

~~~python
json.dumps(
    value,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
~~~

An implementation MUST validate the restricted profile before applying the
reference serialization.

### 4.2 Domain-separated hashes

For a protocol type T and its canonical JSON payload P:

~~~text
object_sha256(T, P) =
  "sha256:" + hex(
    SHA256(
      UTF8("tycho-builder-proposal-v1") ||
      0x00 ||
      ASCII(T) ||
      0x00 ||
      canonical_json(P)
    )
  )
~~~

The payload excludes its own hash field. Different protocol types therefore
cannot share a commitment merely because their JSON payloads happen to match.

### 4.3 Canonical candidate source

The host canonicalizes submitted candidate source before hashing or evaluation:

1. reject invalid UTF-8, byte-order marks, NUL bytes, and unsupported size;
2. convert CRLF and bare CR line endings to LF;
3. append exactly one final LF if none exists; and
4. preserve all other bytes, including comments, indentation, trailing spaces,
   and Unicode spelling.

The candidate source hash is the ordinary SHA-256 of those canonical bytes,
prefixed with sha256:. Source blobs are transported by content reference or
base64, never by object serialization.

### 4.4 Content-addressed blobs

A blob reference contains exactly:

~~~text
media_type
size_bytes
sha256
~~~

The receiver MUST verify size and digest before decoding. A manifest path MUST
be relative, normalized, unique, and free of empty segments, dot segments,
parent traversal, drive prefixes, or symlink semantics.

## 5. FrozenBuildCase

FrozenBuildCase is the complete Builder-visible semantic input.

Version 1 contains exactly:

~~~text
schema = "tycho.frozen-build-case.v1"
evidence
incumbent
obligations
metadata
case_sha256
~~~

### 5.1 Evidence

Evidence is an ordered, content-addressed manifest. Each entry contains a
stable evidence identifier, media type, byte size, blob digest, and provenance
record. Provenance MUST distinguish observations produced by the environment
from model-authored notes or hypotheses.

### 5.2 Incumbent

The incumbent contains:

- canonical source blob reference;
- canonical source SHA-256;
- executable-model schema or ABI identifier; and
- any trusted helper or dependency digests needed to interpret it.

The incumbent digest is the root of every legal candidate lineage in the
session.

### 5.3 Obligations

Obligations contain:

- obligation_universe_sha256;
- an ordered table of obligation definitions;
- a stable identifier and kind for every obligation;
- its evidence or oracle provenance;
- whether it is protected; and
- its result comparison rule.

The obligation table used to evaluate the incumbent and every candidate MUST be
byte-identical. The Builder cannot add, remove, weaken, or reinterpret an
obligation.

### 5.4 Metadata

Metadata contains only Builder-visible task facts such as the build trigger,
focus, bounded budgets, model ABI, and protocol version. Every Builder-visible
field that could influence reasoning MUST be committed by case_sha256.

Host correlation identifiers, timestamps, and accounting records that are not
visible to the Builder remain outside FrozenBuildCase.

### 5.5 Realm-local construction

The host MUST construct the realm value from validated data-only serialization.
Decoding MUST NOT invoke constructors selected by the payload, pickle hooks,
descriptors, imports, or arbitrary code.

The decoded value MUST be recursively immutable:

- records are frozen;
- sequences are tuples;
- maps are immutable;
- byte data are immutable copies; and
- arrays, if supported, are copied and marked non-writeable.

No mutable object is shared with the host. Large blobs MAY remain outside the
language-model context, but they MUST be copied into a realm-local immutable
blob table before Builder-controlled computation begins. Reading that table is
an ordinary in-realm operation, not a callback or lazy request to a host
service. Only blobs already committed by case_sha256 may appear in it.

Retained prior candidates are not present in version 1.

## 6. CandidateSubmission

CandidateSubmission is the only legal mid-session semantic message.

It contains exactly:

~~~text
schema = "tycho.candidate-submission.v1"
case_sha256
declared_parent_source_sha256
parent_bound_patch
candidate_source
provenance
submission_sha256
~~~

Candidate source is a blob reference. Provenance is advisory, data-only text
that records the Builder's stated reason, evidence references, and expected
obligation changes.

### 6.1 Why a declared parent is insufficient

Candidate bytes plus a parent name cannot prove that the bytes were actually
constructed from that parent. A Builder using stale local source could name a
newer parent and produce an indistinguishable message.

Version 1 therefore requires a **parent-bound patch**. The host, not the
Builder, reconstructs candidate bytes from the named parent.

### 6.2 Parent-bound patch

A parent-bound patch is an ordered list of non-overlapping byte edits over the
canonical parent source. Each edit contains:

~~~text
start_byte
end_byte
before_slice_sha256
replacement_blob
~~~

Offsets address the original parent bytes. Edits MUST be ordered by increasing
start offset and MUST NOT overlap. For every edit, the host verifies the digest
of the replaced parent slice. It then applies edits from highest offset to
lowest, canonicalizes the result, and verifies that the reconstructed bytes
match candidate_source.

A whole-file rewrite is represented by one edit spanning the complete parent,
with before_slice_sha256 committing to the entire parent. Thus even a full
replacement remains bound to an exact parent.

### 6.3 Legal parents

A declared parent is legal only if it is:

1. the exact incumbent committed by the case; or
2. a previously accepted and evaluated candidate in the same session whose
   recursively verified ancestry terminates at that incumbent.

The host MUST reject, before semantic evaluation:

- an unknown parent;
- a parent from another case or session;
- a parent with no receipt;
- a patch whose preimage check fails;
- a candidate blob that differs from host reconstruction; or
- a lineage cycle.

Rejected submissions may affect allowlisted audit and accounting fields only.
They MUST NOT enter the receipt ledger or transaction.

## 7. EvaluationReceipt

EvaluationReceipt is an immutable evaluator observation. It contains two
separately committed parts.

### 7.1 Semantic result

EvaluationSemanticResult contains exactly:

~~~text
schema = "tycho.evaluation-semantic-result.v1"
case_sha256
submission_sha256
candidate_source_sha256
parent_source_sha256
evaluator_sha256
obligation_universe_sha256
candidate_load_status
ordered_obligation_results
obligation_outcome_sha256
semantic_result_sha256
~~~

Every obligation result uses a closed status vocabulary:

~~~text
pass
fail
unsupported
error
~~~

Diagnostics that can vary by process path, wall clock, or runtime formatting
MUST NOT enter the semantic result. Stable diagnostic codes and committed
counterexample data MAY enter it when their canonical form is specified by the
obligation.

obligation_outcome_sha256 is the domain-separated hash of only the ordered
obligation results. It permits outcome-equivalence checks without erasing the
candidate, parent, evaluator, and case identity committed by
semantic_result_sha256.

### 7.2 Observation metadata

EvaluationObservation contains:

- semantic_result_sha256;
- evaluation attempt identifier;
- evaluator wall time;
- evaluator cost;
- resource counters; and
- host audit timestamps.

Observation metadata cannot affect candidate eligibility or publication.

### 7.3 Determinism invariant

For version 1:

~~~text
same case_sha256
same submission_sha256
same evaluator_sha256
same obligation_universe_sha256
implies the same semantic_result_sha256
~~~

Timing, cost, and resource metadata may vary.

The evaluator commitment MUST include executable code, trusted helpers,
dependencies, configuration, random seeds, locale, relevant environment
variables, and every other input capable of changing a semantic result.

If repeated evaluation violates the invariant, the evaluator is invalid for
that tuple. The host records a determinism fault and marks the candidate
ineligible. Version 1 does not conceal stochastic evaluation behind repeated
sampling. Supporting it requires a later protocol with an explicit stochastic
contract.

### 7.4 Purity

Evaluation is observational:

~~~python
receipt = await evaluate(submission)
~~~

means "report what the committed evaluator observes for this submission." It
does not retain a candidate, advance a working incumbent, change transaction
state, or publish theory.

## 8. BuilderReport

BuilderReport is the sole final Builder message. It contains:

~~~text
schema = "tycho.builder-report.v1"
case_sha256
nominated_candidate_source_sha256
judgment
evidence_references
open_uncertainties
report_sha256
~~~

The nominated candidate may be null. Every referenced candidate must already
have an accepted receipt in the session ledger.

The report is advisory. It may explain and nominate, but it cannot:

- mark a candidate retained;
- change an obligation result;
- declare an evaluator valid;
- bypass lineage;
- publish theory; or
- direct an environment action.

An invalid or missing final report makes the Builder session incomplete. It
does not authorize publication from partial output.

## 9. Session state machine

Legal semantic states are:

~~~text
NEW
  -> CASE_FROZEN
  -> SESSION_RUNNING
       -> SESSION_RUNNING        accepted submission + immutable receipt
       -> REPORT_RETURNED        valid final report
       -> SESSION_ABORTED        deadline, crash, invalid final message
  -> FINISHING                   host calls TheoryTransaction.finish
  -> FINISHED                    published, unchanged, or withheld
~~~

Only the host advances this state machine. Evaluating a submission leaves the
semantic session state at SESSION_RUNNING.

After REPORT_RETURNED, further Builder messages are rejected. After
SESSION_ABORTED, the host may preserve the audit record but MUST NOT infer a
report or publish a candidate.

## 10. Receipt ledger

The host maintains an append-only, immutable receipt ledger outside the Builder
realm. Each accepted entry binds:

- submission_sha256;
- semantic_result_sha256;
- observation metadata;
- verified parent;
- verified candidate source; and
- ledger sequence number.

Ledger insertion is not candidate retention. The transaction has no mutable
"current candidate" during Builder execution.

## 11. TheoryTransaction.finish

TheoryTransaction.finish(case, ledger, report) is the only publication
authority.

### 11.1 Preconditions

Before considering a candidate, finish MUST verify:

1. every protocol hash and schema;
2. complete lineage to the frozen incumbent;
3. evaluator and obligation-universe identity;
4. evaluator determinism validity;
5. report references;
6. candidate load success;
7. a complete result for every protected obligation; and
8. that the canonical theory is still the exact incumbent frozen in the case.

If the canonical theory changed after case construction, finish returns
withheld_stale_context and performs no theory mutation.

### 11.2 Pareto-safe eligibility

Let the root vector be the incumbent's result over the complete protected
obligation table. A candidate is Pareto-safe only when:

1. every protected obligation is at least as good as the root under its
   declared comparison rule;
2. no protected result is unsupported or error;
3. at least one publication-driving protected obligation is strictly better
   than the root; and
4. the candidate source digest has not already been evaluated in the case.

This comparison is against the frozen root, not merely the declared parent.
A chain cannot hide a regression by introducing it one step at a time.

The **first protected obligation gain** and the **first unique Pareto-safe
candidate** are separate experimental events. In version 1, unique means a
previously unseen canonical candidate source digest; semantic means that it
loaded successfully and received a complete obligation evaluation. Equivalent
obligation_outcome_sha256 values are also recorded so repeated outcomes cannot
inflate yield. Only the unique Pareto-safe candidate threshold is primary.

### 11.3 Selection

The report's nomination is considered only after eligibility is independently
established.

- If the nominated candidate is eligible, finish may select it according to the
  frozen host policy.
- If there is no valid nomination and exactly one maximal eligible candidate,
  the host policy may select it.
- If multiple incomparable maximal candidates remain and policy does not
  resolve them without new Builder authority, finish returns
  withheld_ambiguous.
- If no candidate is eligible, finish returns unchanged.

Selection policy and tie-breaking MUST be committed before the session. Source
size, elegance, or simplicity MUST NOT break a tie unless an explicit
evidence-backed obligation defines that preference.

### 11.4 Durable mutation

After selection, finish uses the transaction's write-ahead publication and
recovery mechanism. It either:

- publishes exactly the selected candidate and records the decision; or
- restores the prior canonical incumbent.

A crash at any point MUST recover to one of those two states, never a partial
candidate or an evaluator-controlled state.

## 12. Unsupported generalization

Replay success establishes only the obligations replayed. It does not establish
unobserved generality.

For example:

~~~text
Observed:
red objects move left
blue objects move left

Candidate A:
all objects move left

Candidate B:
red and blue objects move left
other colors unknown
~~~

If the obligation universe has no evidence-backed oracle outside red and blue,
the evaluator MUST NOT award Candidate A an additional gain for breadth or
simplicity. If the executable-model format exposes a supported-domain claim,
an epistemic-scope obligation SHOULD mark unsupported extension as a failure.
If the format cannot express or inspect support scope, the evaluator MUST mark
that comparison inconclusive rather than call A superior.

This is a claim ceiling: absence of a counterexample is not evidence for an
unmeasured law.

## 13. Semantic noninterference

Let:

- H0 be the initial host state;
- M be the ordered sequence of accepted CandidateSubmission messages;
- R be the accepted final BuilderReport; and
- A be allowlisted audit and accounting effects.

For any two Builder-controlled computations B1 and B2:

~~~text
H0(B1) = H0(B2)
M(B1)  = M(B2)
R(B1)  = R(B2)

implies

semantic_persistent_state(B1) =
semantic_persistent_state(B2)
~~~

The only permitted persistent differences are fields in A, such as timestamps,
resource use, attempt identifiers, and measured cost. A MUST NOT contain or
influence canonical theory, transaction decisions, environment state,
evaluator semantics, future authority, or publication policy.

Equivalently, host-visible persistent state after Builder termination is a
function only of:

1. accepted CandidateSubmission messages;
2. the accepted final BuilderReport; and
3. host-maintained accounting and audit effects.

Invalid messages, scratch files, local mutation, variable contents, failed
imports, exceptions, destructors, finalizers, monkey-patching, and internal
execution order may alter audit data but cannot alter semantic persistent
state.

## 14. Structural authority requirements

The security claim is structural, not prompt-based. Hiding attributes from
documentation is insufficient.

The implementation MUST ensure that the Builder object graph contains no
transitive path to mutable host authority, including through:

- self traversal or object.__getattribute__;
- closures, globals, class attributes, descriptors, or exception objects;
- imports and module caches;
- inherited file descriptors or shared memory;
- filesystem paths, symlinks, or host mounts;
- pickle, marshal, or dynamic object reconstruction;
- subprocesses, forks, or child inheritance;
- network access;
- monkey-patching or module replacement;
- destructors, weak-reference callbacks, or finalizers;
- nested NOOA calls; and
- mutable objects passed by reference.

Attack tests demonstrate implementation coverage. They do not substitute for
the absence-of-authority argument.

## 15. Proof obligations

An implementation is not protocol-conformant until it discharges these
obligations.

### PO-1: Complete commitment

Changing any Builder-visible semantic input changes case_sha256. Changing only
host-private timing or correlation metadata does not.

### PO-2: Realm isolation

FrozenBuildCase decoding yields recursively immutable, realm-owned values and
no authority-bearing object.

### PO-3: Canonical stability

Independent conforming implementations produce byte-identical canonical JSON,
source bytes, and hashes for a shared corpus of valid values and reject the
same invalid corpus.

### PO-4: Parent-bound lineage

No accepted candidate can name a parent other than the exact bytes against
which its patch preimage was validated. Every accepted lineage terminates at
the case incumbent.

### PO-5: Evaluator determinism

Repeated evaluation of an identical committed tuple produces an identical
semantic_result_sha256. A mismatch prevents eligibility and publication.

### PO-6: Receipt immutability

Neither Builder code nor later host operations can alter an existing semantic
receipt or substitute a receipt from another case, evaluator, obligation
universe, submission, or parent.

### PO-7: Exclusive publication

Before finish begins, accepted submissions and receipts leave canonical theory,
transaction working state, and environment state unchanged. Only finish may
publish or restore theory.

### PO-8: Semantic noninterference

Metamorphic sessions with identical accepted messages and reports but arbitrary
internal computational differences produce identical semantic persistent host
state.

### PO-9: Recovery

Interruption at every durable write boundary recovers to either the original
incumbent or one completely validated selected candidate.

### PO-10: Evidence-bounded judgment

The evaluator and selection policy do not reward unsupported extrapolation.
Unmeasured generality is tied, failed by an explicit support obligation, or
reported inconclusive; it is never silently promoted.

### PO-11: Execution-mode parity

Native and CodeAct adapters use the same case bytes, evaluator commitment,
obligation universe, transaction implementation, model, provider settings,
budgets, selection policy, and common task brief.

## 16. Required test sequence

Tests MUST be passed in this order:

1. canonical value and hash conformance;
2. realm-local immutable snapshot construction;
3. authority and noninterference attacks;
4. friendly known repair;
5. coupled repair where the local fix regresses another obligation;
6. unsupported-generalization case;
7. stale or falsely declared lineage;
8. evaluator determinism fault;
9. interruption and recovery at every publication boundary;
10. Native and CodeAct adapter parity; and
11. zero-ARC matched replay.

No live Builder integration is justified before this sequence passes.

## 17. Matched replay contract

The experiment changes one causal variable: continuity of program state.

~~~text
Native:
provider turn
-> bulk Python call
-> serialized observation
-> provider turn
-> bulk Python call

CodeAct:
provider turn
-> cell
-> persistent locals
-> provider turn
-> cell
~~~

Both arms MUST receive:

- byte-identical FrozenBuildCase;
- task-equivalent instructions with a byte-identical common task brief;
- the same pure submission/evaluation protocol;
- the same model and provider settings;
- the same model and evaluator budgets;
- the same evaluator and transaction;
- the same deadline and continuation policy; and
- the same publication policy.

Adapter-specific mechanical instructions MAY explain how to emit protocol
messages. They MUST NOT tell CodeAct that persistent locals are expected to
help, recommend a continuity strategy, or otherwise reveal the experimental
hypothesis. Both common and adapter-specific prompts are preserved and hashed.

### 17.1 Primary outcome

The primary outcome is:

> Total model plus evaluator cost to the first unique Pareto-safe semantic
> candidate.

Secondary observations are:

- cost to first submitted candidate;
- cost to first protected obligation gain;
- wall time to each funnel stage;
- evaluated candidate count;
- candidate yield;
- protected gains per total model plus evaluator cost; and
- eventual publication disposition.

Wall time is secondary because provider latency can vary independently of
reasoning quality.

### 17.2 Interpretation

A CodeAct win supports only this claim:

> For the fixed model, evidence, evaluator, authority, and budgets in the
> matched replay, continuity of program state improved the efficiency of
> converting frozen evidence into externally verified theory.

It does not by itself establish better gameplay, general Builder superiority,
NOOA framework superiority, or safety beyond the tested protocol
implementation.

## 18. Implementation order

1. Implement the four value contracts and canonical hashing.
2. Implement realm-local FrozenBuildCase construction.
3. Implement the pure evaluator and immutable receipt ledger.
4. Implement TheoryTransaction.finish independently of execution mode.
5. Discharge authority and semantic noninterference obligations.
6. Pass friendly repair, coupled repair, unsupported-generalization,
   stale-lineage, determinism, and interruption/recovery tests.
7. Implement Native and CodeAct adapters over the identical protocol.
8. Run the zero-ARC matched replay.
9. Decide whether CodeAct has earned live Builder integration.
