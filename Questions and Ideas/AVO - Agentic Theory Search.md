# AVO — agentic theory search for the Tycho–NOOA refactor

## Judgment

AVO should not be imported as “give the ARC actor more autonomy” or “evolve ARC
agents.” Its best-fitting target is narrower: **the Builder's search over executable
world models**.

That search already has most of the epistemic conditions that make AVO work:

```text
durable public transitions
        +
editable executable theory
        +
automatic replay verification
        +
first-divergence feedback
        +
rollback and immutable evidence
```

The live gameplay loop does not have AVO's forgiving feedback surface. ARC actions
consume a limited budget, may be irreversible, and reveal only partial evidence. The
Builder's offline theory loop is different: edits are cheap, verifier results are
immediate, and failed candidates need not act. Put the agentic variation operator
there while leaving action legality, evidence, cost, terminal truth, and publication
authority outside it.

The strongest implementation thesis is:

> Turn the Builder from a sequence of fresh code-editing passes over one mutable
> `world_model.py` into an agent-directed search over an immutable, scored theory
> lineage, with host-controlled publication and a bounded advisory supervisor when
> the search genuinely stagnates.

This is not yet a gameplay-benefit claim. It is a source-grounded architectural
hypothesis with preserved-boundary tests available before another ARC call.

## What AVO actually contributes

AVO's important move is not merely “the LLM may make several edits.” One variation
step can inspect earlier solutions, consult knowledge, edit, compile, test, diagnose,
retry, revert, and only then return a candidate. A separate supervisor redirects the
search when it stalls. Its evaluated implementation is a single monotone lineage, not
a population:

```text
prior accepted candidates
  -> agent chooses what to revisit and what to try
  -> many private failed attempts
  -> evaluator-qualified candidate
  -> accepted lineage entry
```

Conceptually, this is a learned, stateful local optimizer behind the interface called
“variation.” That distinction matters here because the refactor must decide which
parts of theory search are model judgment and which parts remain host authority.

## Current refactor: already AVO-like in the important place

### The Builder is already an agentic variation loop

`BuilderStrategy.execute` in
`solvers/tycho_nooa/src/tycho_nooa/builder.py` gives a fresh Builder the current
workspace, verifier state, native tools, multiple model calls, and repeated
edit/evaluate/repair opportunities. Every semantic `world_model.py` edit receives
automatic host-generated feedback through `GatedToolExecutor._wm_feedback` in
`wm_probe.py`. The Builder can run arbitrary bounded analyses through `run_python`,
and its report is advisory.

This is much closer to AVO than to one-shot LLM candidate generation:

```text
inspect evidence -> form theory -> edit code -> automatic replay verifier
       ^                                      |
       +-------------- diagnose --------------+
```

Do not rebuild this loop under a new name.

### The evaluator is unusually strong for ARC

The verifier already exposes several crisp signals:

- whether the theory imports and executes;
- exact transition replay accuracy;
- the first divergent transition;
- level-start reconstruction;
- objective status;
- planner satisfiability and operator/key diagnostics.

This is precisely why agentic search is more plausible for executable-theory
construction than for unconstrained gameplay. Wrong theory edits can die locally
without spending an ARC action.

### Evidence and commitment are already separated

The host records public frames before model deliberation. The actor is persistent,
but each Builder is fresh and cannot act. Candidate goals remain provisional until an
independent objective review and explicit actor resolution. A timed-out Builder's
changed theory can now survive rollback as an immutable, noncanonical,
planner-ineligible candidate with its contemporaneous score and evidence frontier.

These are stronger scientific controls than AVO reports. They should be preserved,
not relaxed in imitation of the paper.

## What is missing

### 1. There is retained debris, but not a live theory lineage

`trigger_builder.py` currently retains a changed theory only on a deadline rollback.
The record is excellent as evidence: it carries the model hash, canonical parent,
evidence-frontier commitment, contemporaneous verifier result, and explicit denial of
publication/planner authority. But ordinary intermediate edits and completed Builder
passes do not become one uniform, model-visible lineage.

The raw tool and decision-checkpoint traces may permit forensic reconstruction. That
is not the same as giving the next Builder a small interface for comparing and
continuing prior theories.

The current prompt tells a Builder to inspect `candidates/world_model/index.json`,
but parent selection is cumbersome. A Builder must discover the index, read a blob,
and reproduce it through generic file tools. The matched continuation result shows
that the retained 8/16 theory preserved its advantage over the 0/16 canonical model,
but neither arm improved. Retention is necessary; it is not sufficient search.

### 2. Builder completion is not a real publication gate

During a normal Builder pass, edits land directly in canonical `world_model.py`.
Automatic verification returns feedback, but `FocusedBuilder.build` does not require
the final candidate to dominate or preserve the starting theory before the changed
file survives the pass. Report freshness and outcome repair are enforced; general
theory non-regression is not.

AVO separates arbitrarily many failed internal attempts from the accepted lineage.
The refactor separates timed-out work from canonical state, but not every completed
yet-regressed final theory.

A scalar “higher replay accuracy wins” rule would also be wrong. A theory can preserve
aggregate accuracy while moving its first divergence, break a previously supported
outcome, or gain fit by encoding an incongruent planner key. Publication needs a typed
partial order over obligations, not one headline number.

### 3. The outer trigger repeats; search direction does not adapt

The trigger control fires on `wm_signal` and repeatedly asks a fresh Builder to fix
the first divergence. This is an excellent causal default, but the retained trigger
trajectory shows the limit: many consecutive four-minute passes received essentially
the same dynamics instruction and timed out. Fresh transcripts provided fresh local
attempts, not explicit meta-level redirection.

The current system has object search over theories. It lacks a bounded search over
**where theory search should go next**.

AVO's supervisor is relevant here, but only as an advisory second search process:

```text
object search: which executable theory fits and predicts?
meta-search:   which representation or discriminator should be tried next?
```

### 4. Contemporaneous fit is not forward predictive validity

The current retained-candidate score measures evidence already available when the
candidate was authored. That is the only legitimate score for live retention order,
and preserving it was the right repair. But it measures fit, not generalization.

As later public transitions arrive, old candidates can be evaluated against evidence
that did not exist at their creation. That produces a second, strictly post hoc score:

```text
creation score: fit to evidence frontier at candidate creation
future score:   prediction on transitions observed after candidate creation
```

The first may choose what to continue immediately. The second may revise that choice
later and provides the closest ARC analogue to held-out evaluation. Never rewrite the
creation score using future evidence; retain both with their different authority.

## Implementation order

Only one new implementation should become active at a time. AVO supplies the
direction; the existing promotion ledger determines what has earned work.

### P0 — explicit agent-controlled continuation from a retained theory

This is already licensed by the matched continuation result and is the smallest AVO
transfer.

Give the Builder one deep, builder-only operation:

```text
checkout_theory_candidate(record_id)
    -> verify immutable record and blob commitments
    -> record the current canonical parent
    -> copy the selected body into the private Builder transaction
    -> run automatic typed evaluation
    -> return the evaluation and lineage edge
```

Requirements:

1. The Builder, not a fixed host sampler, chooses whether a retained candidate is
   relevant to the present discrepancy.
2. Checkout changes only the Builder's transactional working theory. It does not by
   itself publish, enable planning, or authorize an action.
3. The trace records `parent_candidate_id`, source frontier, checkout result, and
   immediate evaluation.
4. A failed or timed-out pass restores the prior canonical body byte-exactly while
   preserving newly measured candidates.
5. The existing generic `read_file` path remains available for inspection; the new
   operation removes lossy LLM-mediated copying rather than hiding evidence.

Implement this at the Builder workspace seam, not in actor prompting. Do not add a
general population framework.

Acceptance test: replay the preserved turn-18 packet and show that the Builder can
select the exact 8/16 candidate by record ID, receives its verifier result before any
further edit, and cannot make it planner-eligible without the normal publication
path. Zero provider and zero ARC calls are sufficient.

### P1 — unify every semantic theory edit into a scored lineage

After checkout has one real use, extract the deadline-only candidate functions from
`trigger_builder.py` into a narrow theory-lineage module. Its interface should hide
blob storage, evidence-frontier commitments, parentage, and score parsing:

```python
record_edit(parent_id, body, frontier, evaluation, cause) -> CandidateId
list_candidates(query) -> CandidateSummary[]
checkout(candidate_id) -> CandidateBody
record_future_evaluation(candidate_id, frontier, evaluation) -> None
```

Record a lineage entry after every successful semantic `world_model.py` edit, not
only the last body visible at timeout. Deduplicate by content plus source context.
Preserve failed-import and regressed candidates as inspectable failures; do not offer
them as default parents unless the Builder explicitly requests them.

The evaluator result should be typed at the verification seam rather than recovered
later by regular expressions from model-visible prose. At minimum retain:

- load/execute status;
- matched/evaluated/observed transitions;
- exact transition accuracy;
- first divergence;
- level-start status;
- objective status;
- planner readiness/contract status;
- evidence-frontier commitment.

Acceptance test: one hermetic scripted Builder makes an improvement, a regression,
and a repair. All three candidates remain reconstructable; only the eligible final
candidate may proceed to publication; the model-visible summary distinguishes them
without exposing host paths.

### P2 — add a conservative publication gate

Run each Builder pass as a transaction and make publication an explicit final event.
The host compares the proposed theory with the starting canonical theory on the same
evidence frontier.

Default eligibility should require:

- the candidate loads and the verifier completes;
- no previously exact transition becomes wrong;
- host-confirmed terminal behavior does not regress;
- planner-contract errors do not newly appear;
- either at least one exact transition is gained, the named first divergence is
  repaired, or an explicit evidence-backed exception is recorded.

This is a partial order, not a scalar fitness. Equal aggregate accuracy is not enough
when the residuals differ. Candidates failing the gate remain in the lineage and may
be revisited, but remain noncanonical and planner-ineligible.

Do not let the supervisor, Builder prose, or actor confidence override a mechanical
failed obligation. If exploratory refactors need a temporary regression, keep them on
the candidate branch until repaired.

Acceptance test: start from a theory with two exact transitions; propose one theory
that swaps which transition is wrong, one that strictly improves, and one that breaks
host-confirmed outcome. Only the strict improvement is automatically publishable.

### P3 — forward-score candidates as new evidence arrives

At each new transition, evaluate a bounded eligible set of older theory candidates in
the sandbox and append a future-evaluation record. Preserve the creation frontier and
future frontier separately.

Start with canonical plus the few explicitly retained candidates, not an unbounded
archive sweep. Cap runtime and classify unavailable evaluation as inconclusive.

Acceptance test: construct two candidates tied on their creation frontier but with
different predictions for the next scripted transition. After that transition is
recorded, the future score discriminates them without altering either creation score.

### P4 — stagnation supervisor as a named treatment

Only after the lineage and publication gate exist can stagnation be measured rather
than guessed. Define a supervisor trigger from host-visible facts, for example:

- repeated same first divergence with no Pareto improvement;
- repeated content hashes or AST-equivalent edits;
- two or more Builder deadlines from the same starting score;
- substantial evaluator use with no new exact transition.

When triggered, a fresh bounded supervisor receives an immutable trajectory packet:

- candidate summaries and parentage;
- typed evaluation vectors and residual changes;
- relevant Builder reports and durable hypotheses;
- evidence-frontier commitments;
- the current discrepancy.

It returns two or three orthogonal search directions, each with a smallest offline
discriminator and a predicted evaluator change. It cannot edit files, publish a
theory, invoke the planner, or act. Its report is advisory input to the next Builder.

The first experiment should replay the known 8/16 plateau. Compare:

```text
control:   current fresh Builder from the selected retained candidate
treatment: same Builder plus one precommitted supervisor report
```

Hold model, provider route, evidence, candidate, tools, tool-step cap, timeout, and
publication policy fixed. Measure score gained per model call, first-divergence
movement, distinct semantic candidate count, and deadline completion. Do not use
gameplay success as the first discriminator.

## What not to implement from AVO

### Do not make ARC actions private repair steps

Kernel compilation and benchmarking are cheap and reversible relative to the search.
ARC actions are the environment intervention. The actor must still commit one host-
validated action at a time, and its consequence must remain public authoritative
evidence.

### Do not use monotone replay accuracy as the sole objective

The same observed history used for search can be overfit. Aggregate replay accuracy
can conceal residual substitution, wrong objectives, or a planner-incongruent state
key. Preserve typed obligations and future predictive scores.

### Do not replace the persistent actor with the Builder

The actor owns game-level attention, hypotheses, experiment choice, and action. The
Builder owns executable-theory construction. Combining them would let code-search
success masquerade as objective or action authority.

### Do not add populations, islands, or crossover yet

The current evidence supports one useful retained candidate and one continuation
comparison. A general evolutionary archive would be a platform built before its
second real use. First make a single lineage usable and inspect whether distinct
valuable alternatives actually accumulate.

### Do not make the supervisor authoritative

AVO does not publish enough about its supervisor to copy it faithfully. Here it must
only redirect attention. Host evidence, the verifier, publication gate, and actor
commitment remain authoritative.

### Do not equate persistent conversation with memory

The refactor's durable files, immutable candidates, decision checkpoints, and traces
are more inspectable than an indefinitely growing conversation. Builder persistence
remains a separate execution-substrate treatment; the current evidence supports
repeated setup cost but not its benefit.

## The resulting architecture

```text
ARC host
  owns public evidence, legality, budgets, terminal truth
       |
       v
persistent Actor
  chooses hypothesis, offline probe, Builder request, and one ARC action
       |
       v
Builder theory-search transaction
  chooses lineage parent
  -> edits working theory
  -> receives typed verifier feedback
  -> compares residuals
  -> retries/reverts
  -> submits candidate
       |
       v
host publication gate
  publishes only mechanically eligible theory
  preserves every other candidate without authority
       ^
       |
bounded Supervisor (only on measured stagnation)
  proposes search directions, never truth or action
```

The interface should remain small. Complexity belongs behind the theory-lineage and
publication seams, not in more forms the model must complete.

## Prioritized decision

1. **Implement P0 next:** agent-controlled checkout of the already retained candidate
   inside a nonauthoritative Builder transaction.
2. Then record all semantic edits as one typed lineage and stop scraping verifier
   prose for scientific state.
3. Add the publication gate before giving the search more time or persistence.
4. Add future predictive scoring before making generalization claims.
5. Test a supervisor only against a preserved plateau and only as a named treatment.

AVO's durable lesson for this refactor is therefore not “put the search algorithm in
the model.” It is:

> Let the Builder decide how to search executable theories because it can close that
> feedback loop; let the host decide what counts as evidence, what may be published,
> and what may touch the environment.

