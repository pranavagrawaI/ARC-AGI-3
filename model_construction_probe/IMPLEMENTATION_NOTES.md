# Decisions before production integration

## 1. Verify prediction, not posterior agreement

The old proposed `update(belief, action, observation)` interface is not inherently
wrong. Its ordering and permissions are underspecified. Score an immutable
pre-observation forecast first, then update. Prefer making the successor branches
already produced by `predict` the input to assimilation; otherwise `predict` and
`update` can become two independent transition implementations.

Prediction claims need an observation contract. Beliefs include knowledge,
occlusion history, and hypotheses that the next rendered grid cannot directly
confirm. Never compare full predicted and posterior belief dictionaries as the
primary forecast metric. Independently measured observable events remain the
anchor for diagnostic evaluation.

## 2. Give the host checkable authority, not an epistemic oracle

A host can verify chronological access, type/shape validity, execution, known
observable comparisons, evidence references, and declared dependency instrumentation.
It cannot infer the true meaning of an arbitrary extractor or guarantee which
mechanism caused a mismatch from one ambiguous observation.

A mismatch report should contain established facts plus candidate explanations,
not one mandatory true-cause label. The returned observations may support several
repairs. An explicit `underdetermined` diagnosis is legitimate. A weak extractor
can mimic dynamics errors; bad frame alignment can mimic omitted hidden state.

Operation selection is a fallible control policy. Ground, Contrast, Close, Compose,
and Test are operation categories, not stages every work item must pass through.
Use one active work item with a bounded repair callback before building a full
production-rule interpreter. Log which operation actually led to a verified gain.

## 3. Distinguish work completion from prospective support

A code patch can be implemented and pass all currently available evidence without
new data being available. Do not hold it open indefinitely awaiting an unseen
observation, and do not call its local tests prospective validation.

Lifecycle:

```text
open -> working -> candidate_ready -> prospective evaluation
             -> needs_evidence
             -> budget_exhausted
```

In an offline trial, code edits and candidate selection may use the prefix. The
suffix is sealed until final evaluation. Once suffix feedback is given to a
builder, it belongs to a new visible prefix and cannot be reused as unseen data.

## 4. Preserve joint and temporal uncertainty

The independently allowed fields x in {0,1} and meter in {8,9} do not represent
the same prediction as the joint alternatives (x=1,meter=9) or (x=0,meter=8).
The observation (x=1,meter=8) passes the marginal test but contradicts both models.

Similarly, two alternating initial phases permit increments (1,2) and (2,1), not
(1,1). Keep candidate identity, branch ancestry, and the candidate's evolving
hidden state. Do not treat representational branch pruning as evidence.

A surviving theory is consistent, not exhaustive or proven true. If every theory
fails, create a revision task. Do not restart a level or reset hidden phase as an
automatic response to a model mismatch.

## 5. Candidate identity does not by itself make a policy valid

Suppose one theory puts success behind the left choice and another behind the
right, with identical observations. Having a successful plan inside each theory
is not having one executable plan for an agent unable to distinguish them.

A policy must choose the same action in observationally indistinguishable
histories. The first integration should test fixed action sequences and execute a
short common prefix with monitoring. A contingent plan may branch only on a real
observable discriminator. Choosing actions using a simulated hidden-theory label
is not an executable contingent policy.

## 6. Use deep reasoning to do work that more computation can change

An unresolved issue is not sufficient to trigger indefinite investigation.
Separate: an unattempted representation/repair search, a missing observation,
an inconsequential uncertainty, and a hard execution/integrity error. Send a
bounded reasoning call the evidence and specific artifact to change. When several
models are observationally equivalent on all retained evidence, request a
reachable discriminating probe if one exists. More reflection alone cannot
produce the absent observation.

Cache the unresolved obstacle against model version, evidence frontier, and
decision horizon. Reopening it should require new evidence, a distinct candidate
repair, or an explicit additional search budget. Keep uncertainty management from
becoming another unrestricted inspection loop.

## 7. Real-trace replay experiment

The user's named LS20, S5I5, and CD82 files were not available as mounted trace
files in this session, and a Files content search returned papers rather than
those artifacts. No real-trace result is included.

For each selected episode, preserve raw chronological observations and actions,
typed frame/protocol events, the allowed-prefix frontier, the available source
and note versions, fixed evaluator targets, and a sealed suffix. Remove future
summaries and revised maps from prefix workspaces. Scope control observations to
information available through the real runtime.

Treatment A: current interface and construction policy.
Treatment B: revised predictive interface with the current construction policy.
Treatment C: B plus obstacle-driven operation selection and construction.

All arms receive the same allowed evidence, model checkpoint, tools, and total
inference/tool budgets. C pays for diagnosis and selection inside that budget.
An existing historical run with different context or compute is not a matched A.
Evaluator-only annotations may label the episode; do not tell C the correct
failure cause or seed it with the correct hidden-phase theory.

Use one diagnostic episode to test plumbing, then replicated prefixes and a
reserved set of episodes/games for a competence claim. First establish that the
construction policy can independently choose and execute a useful operation.
An oracle-selected work item can separately diagnose repair capability, but it
must not substitute for the end-to-end C-versus-B result.

Report joint forecast contradictions, per-target coverage, specificity, no-effect
and secondary-effect errors, frozen rollout survival at horizons 1/2/4/8, token
and tool cost, unresolved/censored cases, and scope lost during repairs. Use
paired episode-level comparisons; adjacent transitions from one run are not
independent demonstrations of generality.

Sparse terminal evidence cannot validate an absent success rule. Saved traces
cannot answer an unrecorded action query. They evaluate model construction on
available experience, not whether a new exploration policy would obtain better
experience. After the offline diagnostic succeeds, controlled live interaction is
needed for that latter claim; a full Kaggle run is not the first step.

## Integration target

Implement a real-trace adapter and one model-backed builder callback using the
existing isolated runtime. Keep the native Python work surface and the current
actor. Do not add another agent role, ontology, or mandatory multi-stage workflow
until an observed failure requires it.
