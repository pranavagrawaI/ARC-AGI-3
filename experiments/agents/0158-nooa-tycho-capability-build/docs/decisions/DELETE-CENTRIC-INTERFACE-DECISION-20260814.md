# Delete-centric interface decision — 2026-08-14

## Status

Implemented and offline-tested. No provider or ARC run is authorized or claimed.

## Requirement

The interface must let a model inspect the evidence needed to repair a rejected
artifact, submit only the semantic judgment that changed, and leave mechanical
state conservation to the host. It must remain bounded and must not teach an
object, action meaning, objective, or route.

The executable recovery case is a claim artifact that:

1. uses three analysis calls;
2. still reads the current probe on the fourth call;
3. revises one heuristic without reproducing the prior register or evidence IDs;
4. returns a valid draft while the host retains omitted heuristics and unions
   prior evidence with the current claim and probe.

## Tracer

Before deletion, the existing host and runtime suites passed. This established
that the old analyze → submit path was executable; it did not establish that the
interface was recoverable.

## Overdeletion passes

### 1. Delete the three-call phase boundary

Removing the exploration lockout made the old rejection test fail because the
fourth evidence-read-plus-submission succeeded and left its fallback response
unused. The missing function was bounded generation, not a second language
inside Python. NOOA's existing `max_iterations` already supplies that function.

Minimal recreate: none. The arbitrary constructor-only AST grammar and its
counter were deleted. New checks require evidence to remain readable after
three calls and separately prove that an artifact without a submission still
stops at `max_iterations`.

### 2. Delete the model-authored full heuristic register

Removing the register entirely broke the post-probe protocol with `a claim after
a probe requires a provisional heuristic`. The missing function was durable,
evidence-bound provisional state.

Minimal recreate: the host owns the full register. The model submits only
`heuristic_updates`; omitted entries are retained, prior evidence cannot be
removed, current claim/probe evidence is unioned mechanically, and the host
sets `epistemic_status='provisional'`. The model no longer copies opaque IDs,
unchanged entries, or a status constant.

### 3. Delete all fragmented evidence readers

Removing `frames()`, `current_frame()`, `frame_summaries()`, `transitions()`,
`probe_results()`, and `attempts()` broke the empirical-surface test before a
claim could be returned. The missing function was one inspectable public state,
not six independent access paths.

Minimal recreate: `inspect_evidence()` returns one named snapshot containing
the current frame, visible frame window, full summaries/history, the transition
and probe already matched to the current frame, events, accepted claim, and
legal actions. The raw `artifact_input` binding is deleted from the sandbox
after host-owned constructors capture it.

### 4. Delete fixed-cell prediction

Removing the fixed-cell constructor broke the exact bounded replacement path.
The missing function was a way to predict successor values at selected fixed
coordinates; translation alone cannot express that case.

Minimal recreate: `predict_cell_values(cell_after(...), ...)`. Its name and
prompt now say that coordinates stay fixed and that it does not track an object
or destination. `predict_translate()` remains the distinct constructor for a
selected pattern moving as a unit. The protocol's exact `Replace` record and
verifier did not change.

## Converged responsibility boundary

The host owns iteration bounds, current-record matching, opaque identifiers,
schema validation, full-register persistence, evidence conservation, action
legality, single-action authority, and non-oracular verdicts.

The model owns grouping, object identity through time, causal relevance,
heuristic content and revision, action choice, and prospective action-conditioned
effects. The host does not infer those semantics from a diff or repair them for
the model.

## Consequence for prior runs

The three candidate runs remain mixed model/interface evidence. Their semantic
mistakes are still observations, but their common terminal failures at the old
lockout/register seam cannot cleanly disqualify the models. This change creates
a more discriminating future test; it does not retroactively rescore a run.

## First provider preflight correction

The first v31 provider preflight exposed a defect that the smaller fake evidence
fixtures did not: the sandbox seed serialized the full raw artifact input, then
attempted to delete its name. With a real 64x64 frame the seed partially failed,
costing the models an iteration and sometimes leaving only part of the constructor
surface. This was host failure, not semantic evidence.

The minimal correction deleted the raw artifact binding entirely. Constructors
now capture only the small host-owned values they require: current observation,
parent claim, matched probe, accepted primary mask, and claim ID. The evidence
grid is serialized once, only inside `inspect_evidence()`. The failed preflights
remain preserved and do not qualify a live run.

The corrected preflight then exposed a second host omission without a sandbox
failure: the prompt called iterations bounded but hid the numeric bound. DeepSeek
and Muse were still exploring on call 12. The minimal correction states the exact
per-artifact call budget and says the supplied functions are pre-bound globals,
not importable modules. The cap remains 12 and no game semantics are added.

The first promoted v31 run exposed a third arbitrary seam. Qwen called
`submit_grounding_claim(...)`, stored and printed its valid record, then attempted
commitment work in the same artifact. The name said submit, but the implementation
only constructed a dictionary unless separately wrapped in `return_result(...)`.
No action was issued. The minimal correction makes the first top-level
`submit_grounding_claim(...)` or `commit_contrastive(...)` statement atomic and
terminal, including assignment form, and discards later statements in that cell.
The redundant wrapper remains accepted for compatibility but is no longer needed.
