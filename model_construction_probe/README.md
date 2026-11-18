# Model-construction probe

An isolated, standard-library reference prototype for the prediction and evaluation
boundary of an obstacle-driven model-construction loop. It is not a production
solver, a secure Python execution environment, or a completed A/B/C experiment.

## Run

Requires Python 3.11 or newer. No third-party dependencies or model-service account
are needed for these tests.

```bash
cd model_construction_probe
python -m unittest discover -s tests -v
python demo.py
```

`results/tests.txt`, `results/summary.json`, and `results/witnesses.json` retain the
executed test output and synthetic witnesses. All fixtures are invented. None are
reproductions of LS20, S5I5, or CD82. The alternating-growth example demonstrates
representational semantics; it is not a proposed ground-truth rule for S5I5.

## Files

- `kernel.py`: immutable JSON snapshots, frozen self-contained source, candidate
  lineage, partial observable predictions, assess-before-assimilate, belief replay,
  and frozen open-loop evaluation.
- `construction.py`: a bounded visible-prefix repair driver with typed work items,
  executable retained-transition checks, and a distinct needs-evidence result.
- `fixtures.py`: scripted worlds for testing the host's semantics.
- `tests/test_probe.py`: unit and integration tests of those semantics.
- `demo.py`: explicit counterexamples and a scripted construction episode.
- `IMPLEMENTATION_NOTES.md`: semantic corrections and the real-trace study design.

## The boundary

```text
visible prefix + candidate source
    -> initialize/rebuild belief under that source version
    -> predict branches and observable claims
    -> freeze forecast
    -> receive observation
    -> assess the frozen forecast against evaluator-owned measurements
    -> assimilate observation into surviving candidate states
```

A forecast is a prediction about observable consequences, not a posterior belief.
A newly discovered fact may enter memory without earning retroactive prediction
credit. All surviving candidate states retain their host-assigned ancestry.

The model contains these functions:

```python
initialize(raw_observation) -> list[{"candidate": str, "state": JSON}]
predict_one(candidate, state, action) -> list[{
    "state": JSON,
    "claims": {observable_name: [permitted_values]},
    "unresolved": [dependency_names],  # optional for informative branches
}]
observe_one(candidate, predicted_state, raw_observation) -> JSON
```

Candidate states are arbitrary JSON, not an imposed game ontology. Each branch's
claims are conjunctive; branches are alternatives. Multiple values inside one
field form an allowed set. Correlations across fields must be represented with
separate joint branches. Correlations across steps must be represented in the
candidate's state and transition code, not by independent per-step sampling.

The host performs candidate elimination before calling `observe_one`. If all
candidates are contradicted, it raises `ModelMiss`. It neither resets the real
environment nor resurrects a rejected theory. Observation extraction uncertainty
must be addressed by an observation contract in a real integration; the fixtures
here have independently specified exact measured values.

The `Targets` object defines evaluator-owned finite domains before evaluation.
These are diagnostic measurement targets, not facts supplied by the model to mark
its own answer. For real grids, the evaluator needs an independently audited
projection to the relevant observables and protocol events. The model continues
to receive only allowed raw observations, not privileged annotations.

## Coverage and specificity

For each observable measured by the evaluator, the kernel takes the union of
permitted values across all maintained branches. An abstaining branch contributes
the entire domain. Otherwise an uncertain branch could be hidden behind a narrow
branch to receive misleading specificity credit.

For a finite domain of size D and predicted union size K, the diagnostic
specificity is `log(D/K) / log(D)`. A singleton has specificity 1; the full domain
has specificity 0. Coverage counts targets whose union is a proper nonempty
subset of the domain. This deliberately treats full-domain predictions as no
informative coverage. Missing measurements are excluded and remain explicit in
the observation record. A target absent from the measurement cannot eliminate a
candidate on that target.

This is NOT a proper probabilistic score. It does NOT demonstrate calibrated
confidence. Error, joint compatibility, coverage, width, and cost must be reported
separately. Candidate count is not a probability distribution. Per-field matches
can all pass while no joint branch fits; the overall verdict then contradicts the
forecast. Matching a forecast is evidence of consistency on that observation,
not global correctness or permission to batch actions.

## Construction lifecycle

`WorkItem.operation` describes a kind of work, not a mandatory state-machine stage.
The repair driver takes an already-selected work item and a builder callback:

```python
builder(work, visible_prefix, incumbent, prior_attempts) -> source | NeedEvidence
```

The callback receives no suffix argument. `construct()` checks the proposed source
on the visible prefix and evaluator-specified completion checks. It returns one of:

- `candidate_ready`: passed these visible-prefix checks, not prospective support;
- `needs_evidence`: a named dependency and proposed observable experiment;
- `budget_exhausted`: bounded attempted repairs did not meet the checks.

The default of two candidate attempts is just an editable engineering default.
An LLM adapter must impose its own total token, tool, and time limits. The driver
does not implement a semantic operation selector, infer the ground-truth failure
category, or execute experiments. The demo's builder is explicitly scripted.

Retained-transition checks must cover the relevant previously supported claims.
The driver rejects contradictions anywhere in the replayed prefix and requires
informative matching predictions on named completion checks. It does not infer
all dependencies of arbitrary Python, automatically detect every lost scope, or
prove that an extractor identifies the intended real-world entity.

## Chronological evaluation

`rebuild()` reconstructs the current belief by replaying only a supplied prefix
under the candidate's frozen source. Passing an old-version belief into revised
source is an error. Newly added state fields are therefore initialized from
available evidence rather than silently invented.

`score_one_step()` predicts, assesses, then assimilates each new observation,
without changing source code. `score_open_loop()` constructs the entire fixed-
action forecast tree before inspecting suffix observations. During audit, a match
must have a lineage compatible with all earlier observations. Audit never invokes
`observe_one` or corrects predicted physical state. A candidate contradicted on
an earlier step cannot return to explain a later step.

The open-loop field metrics describe the originally frozen per-horizon prediction
sets, while the joint verdict additionally imposes whole-path consistency.
The result stops at the first contradiction; later horizons are censored, not
successful or independently missing predictions.

A branch-budget overflow raises `BranchLimit`. It must be recorded as a resource
failure or unresolved computation, not as a reason to claim that discarded
hypotheses were falsified. The default cap is 32 and is configurable.

## Deliberate limitations

**Do not run untrusted generated source using this in-process implementation.**
`exec` has ordinary Python authority. Import rejection, frozen dataclasses, and
fresh namespaces do not provide containment against hostile code, introspection,
filesystem access, or nontermination. Use the production solver's isolated realm,
a closed import/dependency bundle, restricted capabilities, and host-enforced
resource limits. Suffix storage must not be mounted in a builder's sandbox.
A Python callback with access to its caller's process is not a secrecy boundary.

For the same reason, forecast hashes provide reproducible identity and ordinary
mutation detection, not authentication against a malicious host or arbitrary
Python execution. The prototype freezes one self-contained source string. A real
version must also bind imported helpers, data tables, configuration, observer
version, and execution environment to the model version. The demo assumes pure
model calls; it does not prove purity or enforce deterministic behavior.

No real trace adapter, LLM invocation, automatic operation selection, resource-
aware planning, general dependency analyser, or contingent-policy planner is
implemented. No game-general performance or C-over-B gain has been measured.
The next empirical test is a real chronological-prefix episode, with identical
observable evidence and inference budgets for B and C.
