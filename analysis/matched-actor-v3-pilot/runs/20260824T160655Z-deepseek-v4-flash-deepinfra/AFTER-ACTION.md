# DeepSeek Matched Actor V3-Pilot after-action

## Outcome

**Scientific status: incomplete; no Native-versus-CodeAct judgment.** The
launcher stopped during the third packet before dispatching a request whose
conservative input bound was 19,399 tokens against the authorized 16,384-token
ceiling. The run made 21 conforming provider calls, spent `$0.003644092`, and
executed zero ARC actions. No violating request was dispatched.

V3 did repair the V2 defect: the Native arm successfully crossed the legacy
query boundary and received one evidence disclosure. The new failure is
cumulative provider-context growth after multiple large evidence results.

## Observations

- `action-failure-illegal-bait`: both arms completed with legal `UP` proposals;
  CodeAct used 2 calls and Native 3.
- `action-success-grounded`: both arms exhausted six calls without a terminal
  result.
- `history-success-level1-noop`: Native made four conforming calls, successfully
  received a 5,938-byte full-frame disclosure, then requested full transitions.
  Adding the next result would have raised the serialized request to 18,887
  bytes; admission rejected it before `CallPlanned` or dispatch.
- The remaining three packets and the CodeAct counterpart for the interrupted
  packet were not exposed.

## Cause

Each evidence page is individually bounded, but Native tool results remain in
provider-visible history. The first full page was safe; the next page plus the
existing transcript and schemas was not. This is a composition failure between
bounded disclosure and bounded accumulated context, not recurrence of the V1
query-class defect.

DeepSeek also chose an expensive path—full frames followed by full transitions—
but that path is admitted by the advertised API and is reasonable for the task.
Engineering for a smaller model therefore cannot rely on it discovering a
metadata-first or minimal-field strategy unaided.

## Decisions

- Preserve the precommitted stop; do not rerun or raise the 16,384 ceiling.
- Do not score the partial pair results or advance either architecture.
- Treat successful Native disclosure as evidence that V3 fixed its intended
  boundary, while treating the pilot comparison as unresolved.
- Before another paid run, add an exact-evidence retrieval form (for example a
  ref/turn selector) or a trusted context-collapse rule, then replay this exact
  preserved trajectory through request admission offline.

## Verification

- 21 calls were planned, dispatched, received, route-validated, and accounted.
- Actual and authorized-accounted spend: `$0.003644092`.
- Violating spend and unknown exposure: `$0`.
- Maximum dispatched input bound: 14,280; maximum actual prompt: 7,188.
- Rejected undispatched request: 19,399 input-token bound, 18,887 serialized
  bytes, request hash
  `sha256:59b459442558e4d0a882250a72ef32a4b41a590edf4bb70bd4787a6ea01447fe`.
- Run state: stopped with
  `request_admission_failure:pre-dispatch input-token bound exceeds authorized per-call cap`.

These facts establish safe containment and the new cumulative-context seam.
They do not establish relative Actor quality.

## Resume

Start from `authorization-events.jsonl` event 112. Build a zero-provider replay
that substitutes one narrow exact-record query or deterministic context collapse,
and require the next request to remain below the existing ceiling with identical
evidence semantics. Any paid retry requires a new version and authorization.
