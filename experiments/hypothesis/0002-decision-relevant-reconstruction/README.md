# 0002 — Decision-relevant executable reconstruction

**Status:** architecture and next-loop contract. No contestant capability claim has
been established by this document.

## 1. Research question

> Can an admissible small-model agent acquire a decision-relevant executable
> reconstruction cheaply enough that deterministic simulation replaces most real-world
> trial and error?

The target is not a complete simulator. It is the smallest reconstruction that changes a
consequential decision: choosing a probe, rejecting an unsafe plan, selecting an option,
or executing a route.

## 2. Governing decomposition

Deterministic and semantic are not opposites.

- **Machinery** performs reproducible computation once a representation, hypotheses,
  objective, and risk contract are specified.
- **Game Judgment** chooses those semantic commitments from incomplete evidence.
- An **Executable Conjecture** is the bridge: a semantic claim compiled into something
  machinery can replay, falsify, and use prospectively.

```text
public observations and actions
        ↓
machinery: immutable evidence and measurements
        ↓
enumerated candidate representations, correspondences, and patterns
        ↓
Game Judgment: select, reject, or propose an interpretation
        ↓
machinery: compile an executable conjecture
        ↓
replay against admissible history
        ↓
Working Theory
        ↓
simulation, search, and budgeted plan comparison
        ↓
Game Judgment: execute a plan, run a discriminator probe, or abstain
        ↓
machinery: release one authorized action when required
        ↓
prediction comparison and mismatch report
        ↓
Game Judgment: diagnose and revise semantic commitments
```

The model does not have to write a simulator or route. It supplies the choices that the
machinery cannot decide from its own enumeration.

## 3. The reconstruction has three separately falsifiable conjecture classes

Treating every model-authored claim as one undifferentiated world-model statement would
make mismatches impossible to diagnose. A useful reconstruction needs at least:

### 3.1 Grounding Conjecture

Names what a state symbol refers to in public evidence and how it is recovered.

```text
G7: candidate track T12 is the controlled entity under representation P3
G9: region family F4 instantiates the same role across levels
```

It owns semantic identity and correspondence. Machinery may enumerate components and
advance a chosen track, but it does not assert that the component is an entity or player.

### 3.2 Transition Conjecture

Names a state change and its conditions.

```text
C4: when G7 enters region G9, latent relation R advances by one modulo four
```

It compiles to an executable transition over reconstructed state. Its replay result is a
mechanical fact about fit to cited history, not proof that the semantics are correct.

### 3.3 Objective Conjecture

Names the condition worth pursuing and its relation to public progress or terminal events.

```text
O2: entering candidate region G11 while relation R equals its displayed value
    causes level completion
```

Planning cannot begin merely because dynamics fit. It needs an objective conjecture and a
risk policy. A terminal correlation may support an objective conjecture; machinery must
not silently promote it to an objective.

A plan must cite all three classes. A failed successor can then be attributed as an open
question over grounding, transition, objective, or execution rather than collapsed into
“the model was wrong.”

## 4. Minimal runtime objects

These are logical objects, not frozen Python schemas. The first prototype should include
only fields exercised by the next-loop verifier.

### Closed Occurrence Records

Immutable public observations and action-conditioned transitions, addressed by run and
step. The existing 0007 raw observation and transition records remain the authority;
request records remain a separate audit trail.

### Representation Specification

A model-selected or enumerated recipe for producing candidate structure from evidence,
for example connectivity, colour treatment, terrain exclusion, temporal window, and
correspondence rule. It is a semantic commitment, not hidden host policy.

Every run carries a representation manifest. For each parameter it records the finite
admissible domain, selected value, provenance (`contestant_prior`, `model_selected`,
`oracle_fixture`, or `implementation_only`), evidence frontier, and freeze status.
Capacity bounds may limit total candidates independent of content; they may not rank,
truncate, or choose candidates by host-estimated relevance.

### Candidate Scene

Reproducible measurements under one representation specification: regions, shapes,
positions, repetitions, and candidate correspondences. Candidate IDs are scoped to the
representation version and evidence frontier; they are not universal object identities.

### Executable conjecture

A typed, runnable claim with:

- stable conjecture ID and class;
- representation version and grounded symbols;
- scope and precondition;
- executable effect or predicate;
- supporting and contradicting evidence references;
- replay result at a named evidence frontier;
- status: proposed, working, contradicted, superseded, or unresolved.

Natural-language rationale may accompany it, but only the executable part enters replay,
simulation, or search.

### Working Theory

The selected compatible set of grounding, transition, and objective conjectures used for
the current decision. Competing conjectures may remain alive outside it. “Working” means
useful enough under the declared risk contract, not known true.

### Plan Certificate

```json
{
  "objective_conjecture": "O2",
  "working_theory_version": "W12",
  "initial_reconstructed_state": "S18",
  "actions": [3, 3, 1, 4],
  "relied_upon_groundings": ["G7", "G9", "G11"],
  "relied_upon_transitions": ["C4", "C9"],
  "predicted_states": ["S22", "S27", "S31", "S36"],
  "unverified_assumptions": ["C9 has not been tested from S27"],
  "risk_disposition": "execute first action only; compare successor before continuing"
}
```

The plan certificate is output from machinery. Game Judgment supplies the objective,
Working Theory, abstraction, and risk disposition that parameterize the search.

## 5. Deep-module seams

Avoid exposing every internal detector and planner primitive. The target can be expressed
through four small interfaces.

### `CandidateWorld.measure(evidence_frontier, representation) -> CandidateScene`

Regenerates candidate regions, tracks, relations, and exact measurements from authoritative
public evidence. It asserts no game role.

### `Theory.evaluate(proposal, evidence_frontier) -> ConjectureReport`

Compiles a typed proposal, replays it over every applicable recorded transition, and
returns exact supports, contradictions, uncovered cases, and state traces. Invalid or
unexecutable proposals fail without entering the theory.

### `Planner.compare(state, theory, objective, budget) -> PlanSet`

Searches the executable reconstruction and returns candidate plans, predictions,
assumption dependencies, and resource/risk summaries. The planner does not choose which
objective or theory to trust.

### `Executor`

`release(decision) -> AuthorizedAction | NoAction`

Releases at most the action authorized by an `EXECUTE` or `PROBE` decision and retains the
prediction and plan suffix needed to evaluate the successor. `ABSTAIN` releases no action.
The executor enforces the decision; it does not choose among execute, probe, or abstain.

`observe(public_successor) -> ExecutionReport`

After a released action produces a public successor, compares that successor with the
retained prediction and invalidates the remaining suffix on mismatch. It reports the
mismatch; it does not decide which semantic conjecture to revise.

The Game Judgment crossing operates over the outputs of these modules. It is not another
peer module that hides its semantic commitments.

## 6. Fit with the current 0007 implementation

### Keep as substrate

| Current 0007 element | Role in the target |
|---|---|
| `bench/record.py` and raw run artifacts | Immutable research/audit record |
| Exact public observation and action crossing | Closed Occurrence Records |
| `bench/state.py` fact/claim/advice separation | Authority invariant; extend claim types rather than merge channels |
| `bench/tracker.py` proposal-then-verification lifecycle | One grounding adapter after Game Judgment names a candidate |
| `bench/retention.py` raw archive and evidence-linked checkpointing | Continuity substrate, after checkpoints carry theory IDs and replay status |
| Action legality, budgets, stop conditions, provider audit | Execution and experimental invariants |
| Five-lane viewer | Human review surface; add reconstructed state, conjecture replay, and plan certificate |

### Replace rather than layer

| Current behavior | Why it does not fit | Target behavior |
|---|---|---|
| Model emits one environment action every turn | Keeps planning and execution inside the model | Model selects/proposes semantic commitments; planner produces actions |
| `goal_hypothesis`, `local_objective`, and `route1` are prose | Cannot be replayed, falsified, or searched | Objective and transition conjectures compile to runnable predicates/effects |
| Exact `RULES.md` is supplied | Oracle semantics; useful only as an upper-bound condition | No task-specific rules in an admissible run |
| Exact grid and full image are repeatedly sent to the model | B.4 shows unreliable raw reconstruction; local hardware makes image use costly | Machinery measures pixels; the model receives bounded candidate structure and may request evidence |
| Sparse supervisor questions audit a draft action | Audits after the model has already done perception and planning | Audit grounding, conjecture, objective, and risk commitments before search/action |
| Edge gate asks the model to classify terrain and route | D produced form compliance without reliable live judgment | Machinery checks geometry under a selected representation; model adjudicates ambiguity or changes representation |
| Retention checkpoint summarizes prose beliefs | Preserves claims but not executable semantics or replay validity | Carry theory IDs, evidence frontier, replay reports, plan, and unresolved mismatches |

The 0007 tracker must not become a universal object detector. Its crucial invariant is
still correct: only Game Judgment proposes a referent, and only subsequent public evidence
can verify its behavior. The target generalizes from one chosen box to versioned candidate
representations and grounded symbols.

## 7. What prior evidence establishes

### Observation

- 0007 B.4: Gemma failed both raw perception and route construction from verified state,
  while largely succeeding when given a verified route.
- Luna and Sol planned and executed from verified state but remained unreliable at raw
  coordinate reconstruction.
- 0007 D: a local edge-check schema produced compliant output but repeated known blocked
  edges under live load.
- The inverted-authority probe recovered transferable object/resource measurements, but
  its goal-relation enumeration degraded on the holdout and model pair selection scored
  0/4.
- Lesson 0049 efficiently searched and replayed plans when given the real LS20 state,
  engine, ontology, and objective.

### Inference

The current per-turn model interface asks one crossing to reconstruct pixels, choose an
ontology, infer dynamics and objective, plan, and execute. Larger models remove some but
not all failures. Deterministic perception alone does not repair Gemma's planning, and a
prose planning field does not make claims cumulative or executable.

### Hypothesis

A small model may be useful if its work is narrowed to Game Judgment over explicit
candidate representations and conjecture families, while replay, contradiction search,
simulation, planning, and execution are machinery.

### What is not established

- that the candidate language can express decision-relevant rules across games;
- that a small local model can propose the missing representation or rule family;
- that replay fit predicts prospective usefulness;
- that an approximate learned reconstruction is faithful enough for search;
- that the architecture reduces charged actions rather than adding ceremony.

## 8. Smallest complete vertical slice

Do not first refactor all of 0007 or build a general symbolic simulator. Separate an
oracle-authored operational fixture from the acquisition probe. They may exercise the same
runtime, but they do not answer the same question and their inputs must not be pooled.

### 8.1 Operational tracer bullet — plumbing only

Use one recorded admissible LS20 level-1 prefix and one held-out transition. A scripted
Oracle Coach fixture may name the known controlled entity, interaction region, latent
cycle, and objective relation in order to drive every runtime seam once:

1. import public frames/actions as Closed Occurrence Records;
2. compile the fixture's grounding, transition, and objective conjectures;
3. replay them over the prefix;
4. search the resulting reconstruction;
5. emit a plan certificate; and
6. compare one frozen prediction with the held-out public successor.

This fixture establishes only that the machinery transports, compiles, replays, searches,
and compares as contracted. Because the fixture contains the LS20 answer, it supplies no
evidence that an admissible contestant can acquire the reconstruction. It must be labelled
Oracle Coach in every artifact and must not define the contestant-visible vocabulary for
the next condition.

No concrete trace or occurrence frontier is authorized by this note yet. Before coding
the fixture, record its exact trace identity, inclusive prefix frontier, held-out successor,
and why that occurrence was selected. Label it a researcher-seen development fixture;
withholding its successor from the runtime does not turn it into prospective evaluation.

### 8.2 Acquisition probe — neutral candidate IDs and frozen grammar

Freeze the evidence prefix by trace identity and step frontier, the candidate generators,
the conjecture grammar, and the held-out scoring projection before collecting a model
response. The model-visible problem contains no `player`, `changer`, `goal`, `rotation`,
or known-good route. It contains only:

- public action and state fields;
- versioned candidate IDs and exact generic measurements;
- transition tables indexed by candidate and action;
- the closed game-general conjecture grammar;
- resource bounds; and
- exact evidence references the model can resolve.

The representation manifest is part of the acquisition authority check. Terrain masks,
thresholds, correspondence rules, and candidate-family settings learned from development
games may appear only as explicitly labelled Oracle Coach fixture values. In the
admissible condition, the host exposes the frozen finite parameter domains through generic
evidence operations and the model selects among them. The host supplies no relevance-
ranked top-k list, fitted semantic threshold, or outcome-selected truncation.

The model must choose a representation, ground symbols, and submit transition and
objective conjectures. Machinery compiles and replays them, then returns candidate plans,
discriminator probes, or `UNKNOWN` to Game Judgment. Game Judgment selects the
consequential decision, frozen as one of:

```text
EXECUTE(plan_certificate_id, first_action)
PROBE(discriminator_id, action)
ABSTAIN(reason_code)
```

The executor validates and enforces the selected decision. It does not promote
`UNKNOWN` to an action or choose among the three decision forms.

The held-out successor scores only the frozen prediction for the historically recorded
action. If the intact and perturbed judgments select different actions, probes, or
abstentions, the fixture records that decision divergence but does not label either branch
better. The successor never generates candidates, fills conjecture fields, or selects the
abstraction used for comparison.

The acquisition interface is still policy-developed on researcher-seen games. A green
LS20 result establishes at most local interface readiness and prospective consequence.
The grammar and candidate policy must then be frozen before a policy-development holdout
can test whether the mechanism transfers.

Before opening that holdout, freeze the fixture-selection rule, representation domains,
conjecture grammar, planner, controls, prediction projection, and acceptance rule. Record
the second trace identity and protected frontier only after the freeze. This establishes a
Policy-Development Holdout, not a Researcher-Blind Holdout unless its mechanics and
outcome were also unavailable to the researchers.

### Closed v0 conjecture language

Start with a closed game-general inventory, acknowledging that choosing this inventory
after studying LS20 makes the first fixture policy-development evidence rather than a
generality test:

- grounded region membership and contact/entry events;
- tracked-candidate position and finite-valued latent variables;
- action-conditioned translation or no-op;
- conditional assignment and modular `advance(variable, modulus, delta)`;
- model-selected equality/inequality relations between observed or latent values;
- conjunction with model-selected region entry or contact;
- bounded unknown result when a required grounding or transition is uncovered.

No arbitrary Python from the model, no free-form executable code, no learned neural
simulator, and no generic DSL extension mechanism in v0. Add a primitive only when an
observed candidate conjecture cannot be expressed and that missing expression blocks the
next decision.

## 9. Next-loop verification contract

### Behavioral claims

Keep the claims separate:

1. **Operational:** given a fixed evidence prefix and Oracle Coach proposal, the runtime
   produces the same replay report, plan certificate, and frozen held-out comparison.
2. **Acquisition:** given only the neutral candidate packet, an admissible judgment adapter
   produces an executable reconstruction whose controlled perturbation changes the frozen
   consequential decision before held-out evidence.

Neither claim answers the research question in §1. The operational claim establishes
plumbing; the acquisition claim establishes reproducibility, controlled sensitivity, and
at most prediction fit for the historically recorded action on a development fixture. It
does not establish that the intact decision is better than a divergent alternative.
“Replaces most real-world trial and error cheaply enough” requires the later matched
package comparison over multiple precommitted decisions and charged interaction.

### Gates

1. **Operational validity:** identical evidence, representation, and proposal produce
   byte-identical candidate state, replay report, and plan certificate.
2. **Authority:** no game identity, source state, private engine value, hand-authored LS20
   role, known-good route, Oracle Coach field, or held-out successor enters the acquisition
   request or its candidate-generation path. Every representation parameter has declared
   provenance; no host-selected terrain mask, fitted threshold, relevance ranking, or
   semantic candidate truncation is permitted.
3. **Replay discrimination:** at least one replay-plausible competitor remains explicit;
   the machinery either exposes a prefix contradiction or emits a discriminator rather
   than accepting the intended theory because it is the only supplied theory.
4. **Controlled prospective consequence:** predeclare one replay-plausible perturbation of
   a decision-relevant candidate measurement and run `intact → perturbed → intact` through
   the judgment interface. Both intact runs must reproduce the same mechanically extracted
   decision signature; the perturbation must change that signature in the predicted way.
   Ordinary schema completion or any output change is insufficient.
5. **Frozen prediction:** before revealing the held-out successor, the plan certificate
   names the exact candidate IDs, fields, canonical values, and comparison function to be
   scored. Agreement means exact equality on that projection; the abstraction may not be
   weakened after seeing the successor.
6. **Historical-action scope:** the held-out successor scores prediction fit only when the
   selected action is the historically recorded action. A different action, probe, or
   abstention is recorded as decision divergence and receives no better/worse label from
   this fixture. No reconstruction-disabled superiority claim enters this gate.
7. **Burden:** record candidate count, serialized bytes, model tokens/time, replay time,
   search expansions, and unresolved cases.

### Raw evidence to inspect

- exact public prefix and held-out successor;
- candidate scenes for every permitted representation setting, with the complete manifest;
- acquisition judgment request and raw response, separately from the Oracle Coach fixture;
- compiled conjectures;
- transition-by-transition replay table;
- rejected/competing conjectures;
- intact, perturbed, and repeated-intact decision signatures;
- plan certificate and simulated state trace with its frozen scoring projection;
- actual-versus-predicted historical-action comparison and any divergent decision
  signatures.

### Falsifiers

- **Representation failure:** neither candidate representation contains stable referents
  sufficient to state the consequential distinction.
- **Authority failure:** removing semantic host labels or the known-good decision makes the
  acquisition condition impossible, showing that the tracer bullet—not the contestant—was
  doing Game Judgment.
- **Language failure:** a concise human semantic claim cannot be expressed in v0 without
  LS20-specific code.
- **Replay failure:** the wrong conjecture fits the same prefix and machinery cannot expose
  a decision-relevant discriminator.
- **Planning failure:** a replay-valid reconstruction cannot produce the expected plan,
  indicating missing state or planner semantics.
- **Judgment failure:** the local model cannot choose/propose a useful conjecture from the
  same candidate state that the scripted fixture can use.
- **Dependence failure:** the intact decision does not reproduce or the targeted
  perturbation does not change the frozen decision signature as predicted.

## 10. Build-up after the first green loop

Each stage is licensed only by the previous stage's raw evidence.

1. **Offline operational tracer bullet.** Oracle Coach proposal, one prefix, one held-out
   transition, no model or live actions; explicitly no acquisition claim.
2. **Judgment-interface probe.** Remove all fixture semantics, freeze the neutral packet,
   perturbation, and scoring projection, then run the small local model. Diagnose
   representation, language, dependence, or judgment separately; make no decision-value
   claim when conditions diverge.
3. **Multiple-prefix replay.** Vary action order and state so a memorized or tautological
   conjecture fails. Preserve counterexamples.
4. **Oracle-qualified same-start branch study.** From one frozen development start, vary
   one selected Representation Specification parameter, grounded attribute, or conjecture
   input at a time and follow each resulting decision through cloned engine state or
   separately realized equivalent starts. Score every branch against the same frozen
   outcome, risk, and burden criteria; allow multiple branches to count as valid solutions.
   Report this as Oracle Coach analysis, not contestant performance.
5. **Matched offline decision batch.** Freeze several decision prefixes and compare
   reconstruction-enabled with reconstruction-disabled judgment under identical evidence
   and model budgets, using the frozen same-start branch criteria and counting every
   passing branch as a valid solution. Primary measures are correct consequential decisions
   and implied probe/action cost; this estimates potential action economy but does not
   measure charged live interaction.
6. **Partial-model planning.** Require the planner to return `UNKNOWN` where the theory is
   uncovered and compare solve-versus-probe choices.
7. **One live prospective action.** Freeze reconstruction and plan before one charged
   action; compare prediction and route mismatch to the recorded contract.
8. **Bounded option execution.** Execute a suffix only while predictions match; invalidate
   immediately on mismatch and return authority to Game Judgment.
9. **Within-game transfer.** Test whether conjectures learned on an earlier level reduce
   acquisition cost on later levels without importing development-run state.
10. **Package comparison.** Compare reconstruction-enabled and disabled conditions with
   identical public evidence, model, and budgets over precommitted decisions and live
   runs. Frozen primary measures are correct consequential decisions, charged acquisition
   actions, avoided probes, total actions, and RHAE. Only this stage can support the §1
   action-economy hypothesis.

Do not stabilize a general architecture, richer ontology, confidence calculus, or
supervisor institution before the vertical loop shows prospective consequence.

## 11. Immediate implementation boundary

Create the prototype as a sibling experiment rather than modifying the working 0007 live
runner in place. Reuse 0007 through narrow adapters for evidence records, run recording,
budgets, and viewing. Reuse mechanisms from `digest_probe.py` only behind a new
candidate-world interface; do not promote its fitted thresholds, top-six truncation,
retrospective win checks, or unstable track IDs into architectural truth.

The first code artifact should stop at an offline plan certificate and historical-action
held-out comparison. It must not call a provider or release an environment action until
the scripted tracer bullet passes and its raw replay table has been inspected.

## 12. Implemented operational tracer bullet

The first implementation is intentionally narrower than the acquisition claim. Its frozen
Oracle Coach fixture is `fixtures/ls20-level1-step13-v1.json`: observations 1–13 of the
named researcher-seen recording are the inclusive prefix, transition 13→14 is held out,
and the fixture records why that occurrence was selected. `reconstruction/` implements the
four offline seams exercised here: closed-occurrence import and candidate measurement,
typed replay, bounded plan comparison, and post-certificate successor comparison. It
adapts 0007's run recorder but does not import its rules, planner, or model-facing packet.

Run from the repository root:

```text
uv run python experiments/hypothesis/0002-decision-relevant-reconstruction/run_tracer.py
uv run python -m unittest discover -s experiments/hypothesis/0002-decision-relevant-reconstruction/tests -v
```

Inspect `closed_occurrences.json`, `candidate_scene.json`, `compiled_conjectures.json`,
`replay_report.json`, `plan_certificate.json`, `held_out_successor.json`, and
`comparison.json` in the printed run directory. Every experiment artifact carries the
`oracle_coach` authority label. The replay intentionally leaves both cycle deltas unresolved
on the prefix and emits their historical action as a discriminator; the Oracle Coach Working
Theory selects one for planning. The held-out successor is loaded only after certification.
A green comparison establishes only deterministic transport and prediction fit for that
recorded action.
