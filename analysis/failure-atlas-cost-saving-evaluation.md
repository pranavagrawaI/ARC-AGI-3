# A cheaper evaluation system for the Tycho-NOOA failure atlas

**Judgment.** Do not answer every failure-atlas question with another paid,
full-game run. Build a ladder in which each claim is tested at the cheapest
fidelity capable of establishing it, and reserve paid closed-loop runs for
questions about policy consequence, level completion, recovery, and transfer.
The largest immediate opportunity is not a cheaper actor model. It is to turn
the existing traces, saved world models, and local black-box game engine into a
corpus of reproducible failure packets, then use exact local branch replay,
metamorphic tests, and bounded decision-point shadows to reject weak changes
before live play.

This is an evaluation design, not evidence that a particular proposed change
works. Where a method is a screen rather than a confirmation, that boundary is
stated explicitly.

## Why the present economics are unsustainable

The two successful ar25 seed-0 runs are useful local replications, but expensive
ones:

| run | total | builder | cached prompt share |
|---|---:|---:|---:|
| [`symmetry-replication`](../experiments/cards/tycho-nooa-symmetry-replication-ar25-20260820T141506Z-RESULT.md) | 215 calls, **$2.3928** | 128 calls, **$1.0819 / 45.2%** | 81.7% |
| [`symmetry-uptake`](../experiments/cards/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z-RESULT.md) | 377 calls, **$4.3270** | 291 calls, **$2.6249 / 60.7%** | 84.7% |

The percentages above were recomputed from the corresponding
`runs/.../llm-trace.jsonl` files. Reconstructing individual fires from the LLM,
builder-tool, and trigger traces shows builder passes of roughly **32--42 model
calls and $0.06--$0.58 each**. The replication result also records that each of
the final two builder passes exhausted 43 tool calls without editing the world
model, and that the actor ignored their advice. This makes builder invocation
the first cost centre to diagnose, but it does **not** license deleting or
truncating the builder: that would create a new Solver Candidate.

Two further boundaries matter:

1. [`llm-trace.jsonl`](../solvers/tycho_nooa/README.md#run-evidence-is-separate-from-the-ordinary-tool-trace)
   records source, model, token counts, cost, and emitted tool names. The current
   artifacts do **not** preserve each exact request body or a restorable
   per-call workspace/history snapshot. A response can therefore be inspected,
   but an old decision point cannot yet be faithfully reconstructed.
2. Reasoning request and response retention are now separate controls. The
   [README](../solvers/tycho_nooa/README.md#run-evidence-is-separate-from-the-ordinary-tool-trace)
   and [`llm_trace.py`](../solvers/tycho_nooa/src/tycho_nooa/llm_trace.py) are
   provider-facing `TYCHO_NOOA_REQUEST_REASONING=1` raises reasoning effort and
   can change retained history; it is a treatment, not passive telemetry.
   `TYCHO_NOOA_CAPTURE_RESPONSE_TEXT=1` only preserves returned text. Freeze and
   record both.

## The claim ladder

| tier | marginal paid cost | appropriate methods | maximum warranted claim |
|---|---:|---|---|
| **0. Static and hermetic** | no game, no model | trace queries; schema and join checks; scripted-client tests; property/state-machine tests; metamorphic transforms | implementation contract, trace integrity, or a failure on an already observed artifact |
| **1. Black-box branch replay** | no paid game, no model | reset local engine, replay recorded action prefix, branch legal next actions, compare the run's saved model predictions | one-step mechanical prediction accuracy at visited states, within the validated engine envelope |
| **2. Frozen decision-point shadow** | model only, no paid game | restore exact request/workspace/history checkpoint; intact-versus-perturbed matched requests; batch independent requests | immediate response dependence at that checkpoint |
| **3. Local closed loop / prefix live** | model plus local engine, or a short paid prefix | validated simulator loop; 1 decision, 3--5 actions, then one level; paired arms | short-horizon behavioral consequence in that game-seed and fidelity only |
| **4. Full live holdout** | full price | paired Mechanism Test on probability-sampled, stratified held-out game-seeds | gameplay effect and, with sufficient coverage and uncertainty, transfer to the declared population |

Passing a cheaper tier is a **promotion rule**, not evidence from the next tier.
In particular, perfect replay does not establish gameplay, and a level win does
not establish general capability. This matches the repository's distinction
between a [Mechanism Test and a Capability Build](../CONTEXT.md).

## Methods and their hard boundaries

### 1. Turn each failure into a replayable packet

The atlas should be a corpus, not only a taxonomy. A packet should contain:

- game, seed, engine/build identity, action prefix, and admissible observation;
- exact model/provider/endpoint and all generation settings;
- credential-free exact request body plus its `Request Payload Commitment`;
- restorable actor history and workspace snapshot/hash before the call;
- response, tool calls, executed action, successor observation, and cost;
- primary theme, interacting themes, predicted failure signature, observed
  signature, validity status, and the packet's claim ceiling;
- Oracle Coach annotations physically separated from Admissible Contestant
  inputs.

The last separation is essential. Source-derived answers or future observations
may grade a packet; they must never enter the actor's prompt or workspace. The
repo already defines [`Oracle Coach`, `Admissible Trace`, and `Request Payload
Commitment`](../CONTEXT.md); the missing machinery is exact checkpoint capture.

**Can establish:** reproducibility, exact matched shadows, regression fixtures,
and cheap re-analysis of one already encountered decision.

**Cannot establish:** what would have happened after a different action unless
the environment is actually branched, nor that the packet distribution
represents unseen games.

### 2. Prioritize black-box local-engine branch replay

This is the most promising zero-model-cost test now available:

1. take a recorded run's game, seed, saved `world_model.py`, and action prefix;
2. reset a fresh local engine and replay the prefix using only its public
   action/observation interface;
3. from that reconstructed state, execute each legal next action on a fresh
   branch;
4. run the **already saved** world model on the same state/action and compare its
   predicted successor with the engine's observed successor;
5. record exact-cell accuracy, changed-cell precision/recall, object-motion and
   termination discrepancies separately; do not let `-1`/unknown cells make a
   false prediction look correct;
6. shrink failures to the shortest prefix/action that retains the discrepancy.

This is not an alien testing strategy imposed on ARC-AGI-3. The benchmark's
official technical report describes known-good recording playback for
reproducibility and a hash-merged state graph whose outgoing edges enumerate
valid actions ([ARC-AGI-3 technical report, sections
3.5--3.6](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)). The
contestant-safe version proposed here is deliberately narrower: use only the
public black-box action/observation interface and keep graph hidden state out of
the packet presented to the solver.

The engine is an evaluator, never actor context. No game source, hidden state,
or future branch observation is exposed to the contestant. Before relying on
the harness, parity-check reset plus prefix replay against held-out released
trajectories and record the intended-use envelope. NASA's simulation-credibility
standard makes the governing point: verification and validation must be judged
against the model's intended use and acceptance criteria, not asserted globally
([NASA-STD-7009B](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf)).

**Can establish:** whether a saved model predicts unchosen one-step branches at
states the solver actually reached; whether errors are local, directional, or
masked by unknown cells; whether a model edit improves a frozen corpus.

**Cannot establish:** that the actor would consult the model, choose the better
action, recover over multiple steps, infer the goal, or transfer to another
game. It also cannot validate branches outside the checked engine envelope.
The official report explicitly says the public demonstration set does not
comprehensively represent the private mechanics and that the private set is
out-of-distribution, so exhaustive public branch replay is still not evidence of
private-set generalization.

### 3. Use trace replay and shadow evaluation only up to first divergence

Ordinary trace replay is valid for deterministic parsers, metrics, protocol
joins, trigger decisions, and analysis of the logged path. Once a changed policy
chooses a different action, its history no longer exists in the log. Stop the
claim there or branch an environment.

Off-policy value estimators can sometimes recover policy value from behavior
data, but they need known action probabilities and adequate action support;
doubly robust methods trade model bias against importance-sampling variance
([Jiang and Li, 2016](https://arxiv.org/abs/1511.03722)). Tycho-NOOA's present
traces do not log a behavior policy distribution over legal actions, and an
interactive deterministic tool policy has poor counterfactual support. Do not
report an off-policy gameplay score from them.

After exact checkpoint capture exists, run **matched local shadows** at a few
load-bearing calls: intact packet versus one precommitted perturbation (remove a
builder report, corrupt the relevant model fact, withhold a completion marker,
or insert a contradictory hypothesis). Keep every other request byte fixed.

**Can establish:** `Local Reconstruction-Content Dependence` -- the immediate
answer or tool choice changes because of the targeted content.

**Cannot establish:** downstream gameplay benefit, generality, or even stable
use at the next turn. A closed-loop paired run is needed for those claims.

### 4. Property-based and metamorphic tests before any model call

The solver already has a scripted `FakeLLMClient` and a cross-interpreter smoke
path. Extend the *test design* around those seams before buying new trajectories:

- stateful rules for observation -> trigger -> builder -> report -> actor ->
  action -> acknowledgement;
- invariants after every step: one legal action, tool-call IDs join uniquely,
  host evidence is monotone, actor and builder lanes remain separate, a failed
  tool does not masquerade as execution, and outcome repair is retried only
  under its declared contract;
- generated sequences of level boundaries, quota errors, malformed advice,
  stale plan hashes, missing files, and worker restarts;
- shrinking to the shortest failing event sequence.

Hypothesis state machines explicitly support generated action sequences,
invariants after every step, and comparison with a simpler reference model
([official stateful-testing documentation](https://hypothesis.readthedocs.io/en/latest/stateful.html)).

For representation tests, derive a follow-up case from a valid source case and
assert a predeclared relationship between outputs. That is the core metamorphic
method described by Chen, Cheung, and Yiu
([paper](https://arxiv.org/abs/2002.12543)). Safe candidates include
serialization round-trips and declared protocol bijections. **Do not assume**
that arbitrary grid-color or action-ID permutations preserve a game's semantics;
first verify the proposed relation through the black-box engine, game by game.

**Can establish:** orchestration, interface, evidence-integrity, and
representation-equivariance failures; cheap counterexamples and regressions.

**Cannot establish:** LLM understanding, exploration quality, world-model
content, or live success.

### 5. Make paid runs sequential, paired, and multi-fidelity

For every intervention, precommit a smallest effect of interest, invalidity
conditions, a bounded primary metric, and a promotion path:

```text
hermetic counterexample
  -> branch-replay improvement on frozen packets
  -> matched decision-point shadow
  -> 3--5 action prefix
  -> one level
  -> full paired game on fresh holdout blocks
```

This is successive allocation: cheap budgets eliminate candidates and only
survivors receive more resource. Hyperband demonstrates the value of this
principle, but also makes it a selection procedure rather than a final
performance estimate ([Li et al., 2018](https://www.jmlr.org/papers/v18/16-558.html)).
Before trusting any low-fidelity score, use a small calibration set to measure
whether it preserves the ordering of full-fidelity outcomes. If it does not,
discard the proxy. Multi-fidelity Monte Carlo can be unbiased only when it still
includes high-fidelity samples and exploits a measured cheap/expensive
correlation; weak correlation destroys the saving
([Peherstorfer, Willcox, and Gunzburger, 2016](https://doi.org/10.1137/15M1046472)).

Within each live game-seed block, compare control and treatment under the same
action cap, observation mode, model, provider endpoint, generation settings,
and time window; randomize arm order. Analyze the within-block difference.
Blocking removes known nuisance variation -- “block what you can and randomize
what you cannot” -- but does not repair a biased game sample
([NIST handbook](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)).
Hosted-model randomness cannot be made common merely by reusing an ARC seed, so
provider/time replication remains necessary.

Use a time-uniform confidence sequence for the paired bounded difference and
stop when it excludes the precommitted practical margin, or when the interval is
narrow enough to declare the result inconclusive. Confidence sequences are
designed to remain valid under continuous monitoring; an ordinary fixed-sample
interval repeatedly peeked at is not
([Howard et al., 2021](https://arxiv.org/abs/1810.08240)). A decisive causal
counterexample can stop failure discovery immediately, but one counterexample
does not estimate prevalence.

**Can establish:** a more precise arm difference with fewer full runs, and an
anytime-valid stop decision under the method's sampling assumptions.

**Cannot establish:** equivalence without a predeclared margin, independence
when repeatedly reusing one game-seed, or transfer beyond the sampled strata.

### 6. Separate diagnostic selection from population estimation

Maintain two queues:

1. **diagnostic queue:** deliberately select traces rich in the suspected
   failure (many builder triggers, sparse observations, long horizons, new
   action vocabularies, terminal evidence). Use it to find and shrink failures;
   make no aggregate performance claim;
2. **estimation queue:** freeze game/seed strata before results, give every unit
   in the declared pool a positive selection probability, record that
   probability, and aggregate with design weights.

Adaptive test selection can remain unbiased only when selection propensities
are recorded and corrected; it reduces variance when its surrogate actually
finds informative cases ([Kossen et al., 2021](https://proceedings.mlr.press/v139/kossen21a.html)).
Stratify by mechanics and evaluation-relevant opportunity, not by a post-hoc
success label: observation sparsity/change rate, action vocabulary, horizon,
trigger opportunities, and atlas theme. Keep a `Policy-Development Holdout` and
periodically use a `Researcher-Blind Holdout` as defined in [`CONTEXT.md`](../CONTEXT.md).

**Can establish:** faster failure discovery in the diagnostic queue; an
appropriately weighted mean over the explicitly declared estimation pool.

**Cannot establish:** the prevalence of a hand-picked diagnostic failure set,
or performance on games with zero inclusion probability.

### 7. Reduce request cost without silently changing the solver

The existing 79--85% cached-prompt shares show that caching is already doing
substantial work; builder **call count and context growth**, not simply cache
activation, are the larger current lever. OpenRouter's official documentation
states that prompt caching is provider/model-specific and uses sticky routing
to retain cache locality
([prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching)).

Neutral or nearly neutral evaluation economies, after parity checks:

- keep identical static system/tool/schema material as a byte-stable prefix and
  append dynamic content; report realized cached tokens and dollars, not an
  assumed hit rate;
- pin model, provider endpoint, quantization, and fallback policy inside a
  matched experiment. OpenRouter otherwise load-balances among providers partly
  by price, and exposes provider and quantization controls
  ([provider routing](https://openrouter.ai/docs/guides/routing/provider-selection));
- batch independent, already-frozen shadow requests when the selected provider
  offers a discounted asynchronous batch API. For example, OpenAI's Batch API
  offers a 24-hour completion window at a 50% discount
  ([official guide](https://platform.openai.com/docs/guides/batch)); this is not
  automatically available through the current gateway and cannot serve a live
  response-dependent loop;
- cache an exact response only for idempotent fixture/smoke work. Reusing it in
  a replication destroys measurement of model variability even if it saves the
  request.

The following are **treatments**, not accounting optimizations:

- lower `ARC_LLM_MAX_TOKENS`, lower tool/call budgets, or early builder cutoffs;
- remove builder fires, reports, or tools;
- trim, reorder, summarize, or compact context;
- turn requested reasoning effort on or off;
- change model, provider endpoint, quantization, or fallback behavior;
- route some actor/builder decisions to a smaller model.

Each may be a good Mechanism Test. None may inherit the old candidate's outcome
claims. Offline truncation can calculate spend *after* a proposed cutoff, but it
cannot prove that the earlier trajectory would have remained unchanged.

### 8. Use smaller models outside the load-bearing policy first

Learned routing can reduce cost in ordinary LLM benchmarks; RouteLLM reports
routers trained on preference data that choose between strong and weak models
([Ong et al., 2024](https://arxiv.org/abs/2406.18665)). That is evidence that
routing is possible, not that a router preserves Tycho-NOOA's interactive tool
policy.

Safe initial uses are corpus triage, deterministic-format conversion, candidate
annotation with audited human/high-cost samples, and proposing cases whose
acceptance is decided by a deterministic verifier. Validate error rates on a
held-out, theme-stratified packet set and preserve uncertainty/escalation.

Putting a smaller model into actor, builder, scribe, or a state-dependent router
changes the Solver Candidate. Promote it only after paired full-model comparison
at the exact call class and a live Mechanism Test. A cheap model agreeing on
prose or action labels does not establish equivalent tool trajectories, model
edits, or downstream success.

## Atlas measurement map

| atlas layer | cheapest discriminating probe | primary measure | claim ceiling before live play |
|---|---|---|---|
| **Seeing**: observation, representation, context | frozen admissible frames plus verified metamorphic relations | entity/change extraction errors; transform equivariance; context item present/absent | representation behavior on packet corpus |
| **Knowing**: memory, world model, uncertainty, goals | black-box branch replay of saved model; prospective predictions scored only after reveal | changed-cell/object/termination accuracy; calibration; contradiction and stale-belief count; terminal-evidence predicate status | model accuracy at visited states and declared branches |
| **Thinking**: exploration, planning, metacognition, tools | matched checkpoint shadows; deterministic plan/verifier audit | probe discrimination, plan legality, expected information gain against enumerated branches, tool/result join, hypothesis revision | immediate dependence and artifact quality, not useful action |
| **Acting**: interfaces, triggers, ownership, discipline | scripted state machine and fault sequences | trigger confusion matrix, recommendation-to-test latency, illegal/malformed action rate, unacknowledged successor count | orchestration correctness |
| **Surviving**: efficiency, stopping, robustness, runtime | trace-cost decomposition plus hermetic fault injection | dollars/call/action, fresh and cached tokens, calls and tools per fire, repeated no-edit passes, recovery after timeout/quota/restart | resource and recovery behavior under tested faults |
| **Learning about the system**: observability, credit, evaluation, provenance | trace-integrity properties plus one-factor paired shadows | missing/ambiguous joins, payload/snapshot completeness, treatment isolation, weighted coverage by theme/stratum | whether later causal claims will be auditable |

For every atlas record, report both the **failure signature** and the **claim
ceiling**. For example, “builder report did not change the next actor tool call
at this restored checkpoint” is valid; “the builder is useless” is not.

## Recommended order of work

### Implemented first slice (2026-08-20)

`tools/failure_atlas.py` now implements steps 2--3 as a bounded screen. Mining
copies the exact decision-phase checkpoint evidence when available, and bundles
the exact predecision Python world-model snapshot by content commitment when a
decision checkpoint exists. Older runs without checkpoints instead receive an
explicitly marked retrospective final-model candidate. The miner also physically
separates successor observations. Replay uses fresh offline engine instances,
requires parity for the entire recorded public prefix (with the online/offline
initial `full_reset` marker difference explicitly recorded), branches simple
legal actions plus only grounded CLICK coordinates, penalizes unknown cells,
keeps variants non-canonical, and identifies the shortest failing prefix among
the selected packets. The implementation does not upgrade its own claim: this
is saved-model mechanics, not policy consequence.

### Implemented second slice (2026-08-20)

`tools/hermetic_gate.py` now implements step 4 as a zero-model, zero-game
promotion gate. Its generated campaign injects twelve stateful corruptions into
a valid trace, requires the declared fault signature from each, verifies two
metamorphic invariances, and greedily reduces the counterexample. The same
auditor applies the contracts to preserved public transcripts, actor/builder
tool joins, trigger-builder lifecycle, planner ordering, and content-addressed
decision checkpoints. A historical symmetry run was rejected for the specific
legacy gap `missing_outcome_checkpoint`; this is evidence that the gate can
exclude an unauditable candidate, not evidence that it predicts gameplay.

### Implemented third slice (2026-08-20)

`tycho_nooa.shadow_eval` now prepares self-contained frozen decision-point
packets from exact call checkpoints and runs capped intact-versus-perturbed
model-only comparisons. A perturbation is bound to the source request and one
exact message value by SHA-256 commitments; preparation rejects any change
beyond that single JSON leaf. Execution checks model/client/provider drift,
randomizes arm order, disables provider retries, journals every sampled call,
and never executes tools or contacts the game. Two or more repeats per arm are
classified as supported only for stable distinct effect signatures, contradicted
for stable equality, and otherwise inconclusive. One live shadow has since
contradicted immediate first-tool dependence on the second reasoning-capture
builder report, but its control did not reproduce the historical first tool and
the provider route was not exposed. The local result is useful; general builder
removal remains unlicensed.

### Implemented fourth slice (2026-08-21)

Paid shadow execution now refuses to start without both an explicit cumulative
cost limit and a per-call reserve. Before each request, the observed spend plus
that reserve must fit. A missing response cost or an observed post-response
overshoot stops every later call and makes the result inconclusive. This is
mechanically enforced client-side containment, not a provider-side billing
guarantee for an already-started request. Non-secret provider-route fields are
preserved when returned; route drift invalidates a causal comparison and absent
route metadata is explicitly marked unverifiable.

The new zero-call `builder-refreshes` miner inspected 102 saved checkpoints
across the two Gemma 4 runs. It found four builder batches costing $0.16460826
in total and two later-report candidates. One is the already-shadowed
reasoning-capture sequence 2 refresh. The only new candidate is feasibility
sequence 3 / actor call 60: 17 builder calls costing $0.06014475 produced a
substantively changed collision report with no action recommendation before the
actor chose `run_python`. This nominates a future matched shadow; it does not
show the refresh was redundant.

1. **Spend no more on broad runs until checkpoints are replayable.** Add exact
   credential-free request commitments and restorable per-call history/workspace
   snapshots to the evidence design. Keep reasoning mode explicit.
2. **Mine current runs into failure packets.** Start with the no-edit 43-tool
   builder passes, the constant `outcome()` goal failures, malformed builder
   advice, terminal checkpoint repair, and the successful symmetry decision.
3. **Build the black-box branch-replay evaluator first.** It has zero model cost,
   directly measures a load-bearing artifact, and can expose discrepancies hidden
   by unknown cells.
4. **Add hermetic stateful/metamorphic gates (implemented).** A candidate that
   breaks trace integrity, ownership, acknowledgement, or protocol invariants
   gets no paid run.
5. **Build frozen decision-point shadows (implemented), then calibrate low
   fidelity.** On a small development set, measure whether branch accuracy and
   checkpoint-shadow effects predict prefix/level outcomes. Delete proxies that
   do not.
6. **Run only one-factor paired promotions.** Prefix first, then one level, then a
   fresh stratified full-game holdout. Use an anytime-valid paired interval and a
   hard dollar/action cap from the experiment card.
7. **Treat builder suppression and model routing as later candidates.** The trace
   makes them economically attractive, but their causal value is exactly what
   must be tested rather than assumed.

The result is not “simulation instead of reality.” It is a system that lets
cheap evidence eliminate claims it can genuinely falsify, while preserving a
small, protected set of real-game runs for the claims only reality can settle.
