# 0006-NOOA competence contract

Status: first executable treatment slice. It licenses the protocol below, not
gameplay competence, a treatment effect, or a claim that the target model uses
the protocol well.

`0002-NOOA` at revision `0364c96` is frozen. The earlier `dfb2d2b` freeze was
reopened on 2026-08-02 to stop the consumed `return_result` call being replayed
as an open tool call, which ModelRun refuses and which blocked every second
action point of this treatment; that directory's `QUALIFICATION.md` records the
defect and why G9 did not catch it.

This directory is the only owner of the competence treatment. It invokes the
frozen worker unchanged; it does not edit that worker, its provider transport,
action schema, sandbox, or host environment API.

## Decision boundary

At each action point the 0006 host:

1. archives the authoritative public observation as a byte-identified
   occurrence;
2. mechanically evaluates the preceding committed prediction, if any;
3. assembles a fixed context packet; and
4. appends it to the frozen model observation.

The model still chooses the action and action data through the frozen
`return_result({"action", "data", "reason"})` channel. For 0006, `reason`
must be a compact JSON object containing exactly:

```text
working_state, test, claim_ids, ledger, assertions,
evidence_requests, reconciliation
```

It must not repeat or override `action` or `data`. The host validates the
action using the frozen legality/coordinate checks, validates the commitment,
then releases exactly one action. A malformed commitment releases no action.

`reconciliation` is the one field the model may omit, and only while no
mismatch is unresolved — in that state `null` is its sole legal value, so
accepting absence supplies no semantic content the model did not author. While a
mismatch is outstanding the field carries real dispositions and stays mandatory.
This is a syntax allowance, not a repair: the host still neither guesses nor
corrects a malformed or semantically invalid commitment.

Rejections are of two kinds, and the host records which. A **precondition**
failure is one no commitment could satisfy — an observation arriving with the
wrong dtype, a spent budget, a closed worker — and it fails fast. A
**postcondition** failure rejects a value the model itself authored: an
unparseable `reason`, an illegal action, a commitment the contract refuses.
Those raise `ModelCorrectableError` and are reported in the run summary as
`correctable`.

A postcondition failure is answered by a **repair crossing**. The frozen worker
refuses a repeated `(run_id, action_point_id)` and validates nothing else about
that integer, so 0006 allocates it from a crossing counter rather than from the
environment step: a repair is simply the next id, and the frozen worker, its
transport, and its persistent Python workspace are untouched. The model's
workspace survives, so the notice names the error and points at the `reason`
string the model already serialized there. It supplies no field values and no
corrected payload. `MAX_COMMITMENT_REPAIRS` bounds the attempts; each one also
spends the frozen spec's model-call budget, which remains the ceiling on cost.

Refused crossings are recorded in full — request, crossing, and rejection — and
the verifier requires each to carry a distinct id, to be marked `correctable`,
and to precede the admitted crossing for the same step. A refusal with no later
admission for its step fails. The count reaches the run summary as
`refused_crossings`, because how often the model must be told to re-author is a
direct measure of whether this contract is usable.

## Admitted mechanisms

| Missing function | 0006 implementation | Host authority | Model-visible difference |
|---|---|---|---|
| Exact before/after evidence | append-only observations, an action-bound adjacent transition occurrence, and a `transition` derivation | records public bytes and computes layer-indexed changed-cell counts; does not label game objects or relevance | model requests a committed `action_id`; the resulting citable derivation appears in the next fixed packet |
| Revisable beliefs | claim create/revise/retire ledger | validates identity, append-only history, and bounds; never judges a claim's meaning | only model-selected active working claims appear in the packet |
| Falsifiable action use | one-action assertions and successor evaluation | evaluates only declared public predicates | previous result/mismatch is visible before the next action |
| Repair after failure | explicit disposition for every mismatch | requires coverage, not a particular disposition or belief | unresolved mismatches block another commitment until addressed |
| Bounded continuity | deterministic context assembly | fixed order, four model-selected working claims, four retained anomalies, four evidence records, and a 12,000-byte cap; no retrieval, ranking, or summary | current occurrence, working state, selected claims, result, mismatch, retained anomaly, continuity, and selected evidence |

The available assertions are deliberately generic: changed-cell count, public
state, and completed-level count. The first evidence operation is deliberately
narrower than the full 0006 v0 frame-comparison catalogue: it returns only
layer-indexed changed-cell counts and public status for one committed adjacent
transition. Neither it nor the assertion set teaches the model what a player,
wall, target, objective, route, or success strategy is.

## Tycho's role

Tycho at local revision `f68912a764372ead0a610db2e1c011d41ce5197e` is an
oracle for candidate mechanisms, not a runtime dependency. Its local
`/home/hipra/projects/tycho/docs/ARCHITECTURE.md` world-model / verifier /
planner separation suggested the useful distinction here:
model-authored candidate state, host mechanical verification, then another
model decision. `0006-NOOA` retains only the narrow verifier-shaped boundary.
It does **not** import Tycho's builder, planner, objective inference, route
search, specialist roles, background reflection, semantic skills, or automatic
context retrieval.

## Audit and comparison

The controller records complete public observation occurrences, action-bound
transition occurrences, derived evidence, commitments, prediction results,
context assembly, ledger operations, and reconciliations in `competence.jsonl`.
Every occurrence/derivation ID is a domain-separated, full SHA-256 commitment
over the contract's integer-only canonical JSON domain. The successor records
and prediction result are flushed during `observe`, before another model call
or a final episode close. The host process log records the competence-packet
digest, worker crossing, commitment admission, frozen action validation, and
successor transition. The frozen 0002 run remains the unchanged control for a
later matched comparison; this slice has not run that comparison.

The executable verifier is the offline end-to-end test: it crosses a real
frozen NOOA worker, records a typed commitment, admits one action, forwards a
multi-layer successor, and exposes the mechanical prediction result at the
next action point.

What that verifier proves, and against which attacker, is pinned in
[THREAT_MODEL.md](./THREAT_MODEL.md). It bounds review: a proposed finding must
name a declared attacker capability, and a finding requiring coordinated rewrite
of the whole artifact is closed by that document rather than fixed.
