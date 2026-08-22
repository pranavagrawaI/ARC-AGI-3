# Questions and Ideas

**Repository:** `arc-agi-3-tycho-nooa-refactor`  
**Snapshot:** 2026-08-21  
**Purpose:** Living ledger for hypotheses, questions, findings, experiment ideas,
and strategies. This document records what the evidence currently licenses and
what would change the judgment. It is not a backlog in which every idea must be
implemented.

## How to use this ledger

For every consequential addition, record:

1. the question whose answer would change a decision;
2. the current hypothesis and a credible rival;
3. the observable result that separates them;
4. the cheapest fidelity capable of producing that result;
5. the invalidity and stop conditions;
6. the evidence obtained and its claim ceiling;
7. the resulting action: reject, retain, revise, or promote.

Use these statuses:

- **Answered locally:** sufficient for the stated local decision, not general.
- **Supported hypothesis:** evidence favors it, but a consequential rival remains.
- **Open:** no discriminating evidence yet.
- **Blocked:** the required measurement is unavailable.
- **Retired:** answering it would not presently change a decision.

---

## Current whole-system judgment

We have answered many questions about whether the **experimental system works**,
several questions about **local solver mechanisms**, and one important question
about the architecture's present imbalance.

We have not answered the ultimate question:

> Can this architecture, using Qwen 3.8 27B, reliably turn unfamiliar
> observations into efficient goal-directed behavior across games?

The strongest current whole-system hypothesis is:

> The solver allocates enormous effort to constructing and verifying transition
> dynamics, but has a weak control loop for turning uncertainty about the
> objective into a discriminating experiment. Planning is consequently starved
> of satisfiable goals, while the builder either repeatedly repairs mechanics or
> goes silent after recorded-path mechanics become accurate.

This is strongly supported on `ar25`. It is not established across games.

### Current causal map

```text
observation
    ↓
state and object representation          partially demonstrated
    ↓
transition-model construction            often strong on the observed path
    ↓
candidate objective formation            candidates appear in prose
    ↓
objective-discriminating experiment       recurrent weak link
    ↓
verified terminal predicate               usually absent
    ↓
planning                                  frequently starved by empty goals
    ↓
action execution                          operational, but expensive
    ↓
terminal evidence and consolidation       repaired offline; live validation pending
```

The system contains two cognitive loops:

```text
FAST LOOP: actor → investigate → act
SLOW LOOP: trigger → builder → model/report → actor
```

The fast loop has demonstrated a same-game, same-seed level completion after a
symmetry comparison converted a hypothesis into a calculated route. The slow
loop has produced useful mechanics, but it can also consume 45–60% of cost,
repeat work after the actor has a route, become silent when only the goal remains
unknown, and threaten the decision deadline. The central problem may therefore
be **allocation of cognition**, not merely insufficient model intelligence.

---

## Questions answered by current evidence

### A1. Can exact model-call decision points be preserved?

**Status:** Answered locally — yes.

Current runs preserve credential-free request bodies, responses, commitments,
and workspace snapshots. Older runs without decision checkpoints cannot be
faithfully reconstructed at the historical decision.

**Evidence:** `solvers/tycho_nooa/src/tycho_nooa/decision_checkpoints.py` and the
Gemma checkpoint runs under `runs/`.

### A2. Can corrupted or incomplete evidence be rejected without model calls?

**Status:** Answered locally — yes.

The hermetic gate detects trace, lifecycle, ownership, tool-join, checkpoint,
and completion-evidence failures. A pass establishes evidence integrity only;
it does not establish solver intelligence or gameplay benefit.

**Evidence:** `tools/hermetic_gate.py` and
`analysis/tool-funnel/20260820-latest-symmetry/hermetic-audit.json`.

### A3. Can saved world models be examined without new model or game calls?

**Status:** Answered locally — yes.

Branch replay restores parity-checked local engine states and tests legal next
actions. It measures saved-model mechanics, not actor use, policy consequence,
or transfer.

**Evidence:** `tools/failure_atlas.py` and
`analysis/tool-funnel/20260820-latest-symmetry/derived-metrics.json`.

### A4. Is recorded-path simulation accuracy sufficient evidence of a complete model?

**Status:** Answered locally — no.

The ar25 model reproduced observed directional transitions, but counterfactual
branch replay found 13 failures among 36 scored legal branches, including
systematic `SPACEBAR`, `UNDO`, and some `RIGHT` errors. Recorded-path accuracy
and legal-action completeness must remain separate measures.

### A5. Can exact content dependence be tested without executing tools?

**Status:** Answered locally — yes.

Frozen decision-point shadows can alter one committed request leaf and measure
immediate response dependence. They do not establish later tool use, gameplay
consequence, or transfer.

**Evidence:** `solvers/tycho_nooa/src/tycho_nooa/shadow_eval.py`.

### A6. Did the current sequence-2 Gemma builder report change the fresh immediate action?

**Status:** Answered locally — no observed effect in the four-call shadow.

Both control and treatment chose `take_action(UP)` twice. However, neither fresh
control reproduced the historical `run_python` first call, provider route was
not recorded, and earlier builder-shaped history remained intact. This does not
license global builder removal.

**Evidence:**
`analysis/shadows/seq2-current-builder-report-gemma4-31b-20260821/AFTER-ACTION.md`.

### A7. Can shadow spending be mechanically bounded?

**Status:** Answered locally — yes, client-side.

Unmetered execution is rejected. A per-call reserve must fit before a request
starts; missing response cost or observed overspend stops later calls. One
already-started response can still exceed its reserve, so this is not an
absolute provider-side billing guarantee.

### A8. Can provider-route drift be preserved and detected?

**Status:** Answered conditionally.

Non-secret route fields are retained when the response exposes them. Observed
route drift makes a shadow inconclusive. Hidden routing cannot be reconstructed,
and old responses cannot be repaired retroactively.

### A9. Can Gemma 4 31B operate the native Tycho-NOOA loop?

**Status:** Answered locally — yes.

The five-action LS20 feasibility run produced valid native tool calls, completed
normally at its action cap, and passed its evidence audit. This establishes API
and solver compatibility, not gameplay competence or equivalence to Qwen.

**Evidence:**
`runs/tycho-nooa-gemma4-31b-feasibility-20260821T073930Z/AFTER-ACTION.md`.

### A10. Can reasoning request and response capture be separated?

**Status:** Answered locally — yes.

The reasoning-capture run retained nonempty reasoning on 37 of 38 calls. Asking
for reasoning changes the model request and remains a treatment; capturing
returned text is telemetry.

### A11. What directly caused the reasoning-capture timeout?

**Status:** Answered locally.

The actor selected `DOWN` with 27.6 seconds remaining, but the solver withheld
the accepted action for a mandatory beliefs-update model call. That call crossed
the host deadline after the builder had consumed 453 seconds. The direct cause
was action withholding; the architectural exposure was an indivisible deadline
shared without protected actor and return reserves.

**Evidence:**
`runs/tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z/TIMEOUT-FORENSICS.md`.

### A12. Is context capacity the demonstrated ar25 bottleneck at 20 actions?

**Status:** Answered locally — no.

Three-grid retention bounded the dominant context-growth term, the actor ended
near 53% of its configured window, and reasoning/tool transmission was healthy.
This says nothing about 64-action or multi-level context behavior.

### A13. Does the goal-satisfiability gate work?

**Status:** Answered locally — as a diagnostic, yes; as a lever, no.

It prevented six pointless searches and correctly identified an unsatisfiable
goal, but six explicit demands produced no candidate terminal predicate. More
planner search is not the first remedy when the objective is empty.

### A14. Can an appropriate representation change gameplay?

**Status:** Answered locally — yes on ar25 seed 0.

The actor used colour-invariant symmetry, established congruence, calculated a
route, executed it, and completed level 0. The completion replicated under the
same game, seed, and broad configuration with a different trajectory. Transfer
is untested.

**Evidence:**
`experiments/cards/tycho-nooa-symmetry-replication-ar25-20260820T141506Z-RESULT.md`.

### A15. Is the builder either globally necessary or globally useless?

**Status:** Open — neither conclusion is licensed.

Early builder passes sometimes changed the model. Later passes sometimes spent
43 calls without an edit, did not cause the winning suffix, or produced a report
with no detectable immediate marginal effect. Builder value must be measured by
phase and decision class.

---

## Priority hypotheses

### H1. The dominant bottleneck is the candidate-goal-to-probe transition

**Status:** Supported on ar25; open across games.

**Current support:** Goal hypotheses repeatedly appear in reports and actor
reasoning, but no objective-discriminating action is selected. Dynamics receive
dense mismatch feedback; provisional objectives have no equivalent positive
evidence channel.

**Credible rival:** Perception or transition modelling is wrong before objective
testing becomes meaningful.

**Cheapest discriminator:** Mine every existing checkpoint into an epistemic
funnel and count the earliest failed transition. Then use branch replay to mark
whether mechanics were adequate at the same point.

**Would falsify or weaken H1:** Most decisions fail earlier at state extraction
or mechanical prediction, or candidate goals reliably produce discriminating
actions without improving outcomes.

### H2. Explicit world modelling can displace actor ownership of purpose

**Status:** Open and consequential.

**Current support:** In the void five-action `no_world_model` prefix, the actor
regrew modelling behavior and interpreted the move counter as a deadline. A
builder-informed actor in another run inverted it into a progress bar. This is
an n=1 lead, not a finding.

**Credible rival:** The contrast is stochastic or caused by different histories,
not by architecture.

**Cheapest discriminator:** Frozen matched checkpoints comparing the normal
actor context with removal of the model/report block, followed by a short
closed-loop promotion only if the immediate diagnostic differs.

**Decision affected:** Whether the architecture should preserve separate actor
and builder ownership, use a single model-building actor, or restrict builder
authority to verified repairs.

### H3. Builder value is strongly phase-dependent

**Status:** Supported descriptively; causal estimates mostly open.

**Prediction:** Cold-start construction and genuine divergence repairs sometimes
help; repeated refreshes after route commitment and dynamics convergence have
low or negative marginal value.

**Cheapest discriminator:** Stratify existing builder batches into model creation,
prediction repair, goal work, planning work, terminal repair, and unchanged
refresh. Join edits, report novelty, actor uptake, cost, latency, and existing
shadows. Select one checkpoint from each important stratum for matched removal.

**Would falsify H3:** Similar causal effects and cost-benefit ratios across phases.

### H4. Correct artifacts are often not causally used

**Status:** Supported descriptively; not quantified.

**Prediction:** World-model or report quality will correlate weakly with action
quality unless the actor performs an explicit evidence-dependent operation.

**Cheapest discriminator:** Join branch accuracy, report commitments, model edits,
and actor tool choices at exact checkpoints. Shadow a relevant fact only where
the actor could use it.

### H5. The terminal-evidence lifecycle is repaired in code but unproven live

**Status:** Supported offline; live confirmation open.

**Prediction:** Host-confirmed completion will create a durable outcome
checkpoint even when no subsequent actor turn occurs, and failed repair will
preserve the win for retry.

**Cheapest discriminator:** First run a scripted actor over a known winning route
against the local engine. Use a paid model only if the scripted lifecycle passes
and model-specific behavior is still the live uncertainty.

### H6. Deadline allocation prevents advisory work from consuming the action path

**Status:** Supported offline; provider cancellation behavior open.

**Prediction:** A builder exceeding its slice is cancelled or abandoned while the
actor retains its protected reserve and returns an accepted action before
model-dependent bookkeeping.

**Cheapest discriminator:** Scripted never-returning and slow clients, followed by
one bounded cheap-provider decision only if real HTTP cancellation remains
consequential.

### H7. Gemma can be a useful screening proxy for Qwen

**Status:** Open.

**Prediction:** Gemma and Qwen agree on the experimental decisions used to
promote or reject mechanisms, even when their prose and exact actions differ.

**Credible rival:** Their tool use, goal formation, or treatment responses differ
enough that Gemma selects the wrong solver changes.

**Cheapest discriminator:** Run both models on a small stratified set of identical
frozen packets. Compare candidate-goal presence, discriminating probe, tool
choice, plan legality, and direction of perturbation effects—not prose similarity.

### H8. Symmetry support transfers beyond one game-seed

**Status:** Open.

**Prediction:** The actor will select an appropriate quotient and use the result
to alter a probe or route in new layouts or mechanic families.

**Cheapest discriminator:** Synthetic and recorded frozen boards with known
transformations first; different seed next; different game only after those pass.

---

## Important open questions

### Q1. Where does the first consequential failure usually occur?

Possible stages:

1. observation/state construction;
2. representation choice;
3. candidate objective formation;
4. discriminator selection;
5. probe execution;
6. hypothesis revision;
7. transition or terminal modelling;
8. evidence uptake;
9. planning;
10. action release;
11. terminal consolidation.

The current atlas names failure themes but does not yet count this funnel across
the corpus. Until it does, frequency and leverage remain impressionistic.

### Q2. What makes a provisional objective create an experimental obligation?

The current architecture permits a goal hypothesis to remain prose indefinitely.
Should a candidate record contain:

- supporting and contradicting evidence;
- predicted terminal signature;
- cheapest safe discriminator;
- deadline in actions;
- proposed/tested/rejected/promoted status?

Would requiring a discriminator improve action choice, or merely create another
ritualized report?

### Q3. When exactly does builder work earn its cost and latency?

Measure value separately for:

- cold-start model construction;
- repair after real prediction divergence;
- objective discovery;
- planning repair;
- terminal consolidation;
- unchanged refresh after route commitment.

### Q4. Does a better world model change the actor's decision?

Distinguish:

- correct model, actor ignores it;
- incorrect model, actor relies on it;
- correct model, actor independently chooses the same action;
- report changes reasoning but not first action;
- report changes a later probe, plan, or recovery.

### Q5. Is the actor-builder separation beneficial?

Would a single persistent scientist preserve ownership better? Would it instead
lose independent verification? Could the builder be restricted to candidate
state production, with the actor retaining promotion authority?

### Q6. Which representation failures dominate on LS20 and other games?

Do errors arise in raw perception, object binding, coordinate frames, graph
construction, or route calculation? Image, text-grid, and host-derived object
representations must be compared as treatments rather than assumed equivalent.

### Q7. Once a goal exists, are forward operators and planner keys sound?

This question becomes relevant only after a satisfiable goal and usable dynamics
exist. `UNDO`, counters, recovery actions, and observationally equivalent states
are known hazards. Do not collapse them before their semantics are validated.

### Q8. Do deadline and terminal repairs survive the real provider lifecycle?

Offline cancellation proves control flow, not prompt provider cancellation.
Terminal repair tests prove the code path, not that a live builder will prioritize
and correctly encode the evidence.

### Q9. Is Gemma a valid development proxy for the Qwen submission model?

Agreement must be defined against experimental decisions, not response wording.
If the proxy fails, Gemma remains a separate cheap candidate rather than a source
of conclusions about Qwen.

### Q10. What transfers?

Separate:

- same game, same seed reproducibility;
- new seed;
- new layout or colour assignment;
- new transformation;
- new mechanic family;
- multi-level adaptation;
- architecture-level transfer;
- private-game performance.

---

## Efficient strategy for answering the open questions

### Tier 0 — Zero-call corpus and contract analysis

Use:

- checkpoint mining;
- trace and lifecycle audits;
- builder-cost and report-novelty joins;
- scripted clients;
- state-machine and metamorphic tests;
- exact source and evidence commitments.

**Maximum claim:** implementation contract, evidence integrity, and failures in
already observed artifacts.

### Tier 1 — Zero-model branch replay

Replay all legal actions from parity-checked saved states. Keep recorded-path
accuracy, legal-action coverage, terminal accuracy, unknown coverage, and
shortest counterexamples separate.

**Maximum claim:** saved-model mechanical accuracy in the validated engine
envelope.

### Tier 2 — Frozen model-only shadows

Select active checkpoints representing distinct causal phases. Alter one exact
content or request factor and execute no returned tool. Begin with two repeats
per arm; expand only when within-arm variability prevents judgment. Require:

- explicit cumulative cost limit;
- conservative per-call reserve based on historical call class;
- provider-route capture when available;
- a positive control;
- precommitted effect signature and invalidity rule.

**Maximum claim:** immediate response dependence at one committed checkpoint.

### Tier 3 — Short closed-loop prefix

Promote only shadows that survive. Run three to five actions from a controlled
state, stopping at the first policy divergence or declared evidence event.

**Maximum claim:** short-horizon behavioral consequence in that game, state, and
model configuration.

### Tier 4 — One-level promotion

Use only for mechanisms that require consequence over several actions: objective
testing, plan execution, terminal recognition, or recovery. Audit evidence before
interpreting outcome.

### Tier 5 — Held-out Qwen evaluation

Reserve Qwen and untouched seeds/games for candidates that passed cheaper tiers.
Progress through:

```text
zero-call artifact test
    ↓
four-call frozen shadow
    ↓
three-to-five-action prefix
    ↓
one level
    ↓
different seed
    ↓
different game
    ↓
Qwen holdout
```

Passing a tier is a promotion rule, not evidence from the next tier.

---

## Proposed next instrument: epistemic-funnel miner

Before another paid shadow, build a zero-model-cost miner that emits one row per
recorded decision with:

```text
run / game / seed / sequence
observation and checkpoint commitments
candidate objective present?
discriminator named?
discriminator executed?
hypothesis revised?
target encoded?
model recorded-path accuracy
model branch accuracy and legal-action coverage
planner goal satisfiable?
plan attempted and valid?
builder phase, calls, cost, latency, edits, report novelty
actor first tool, action, and evidence relation
terminal consequence and consolidation
earliest broken funnel transition
primary and interacting failure-atlas themes
claim ceiling
```

The first output should answer:

1. Which stage fails first most often?
2. How much actor and builder spend accumulates before that failure?
3. Which failures repeat across ar25 and LS20?
4. Which checkpoints best discriminate between goal-mechanism, handoff,
   model-fidelity, and evidence-use failures?
5. Which one-factor shadow has the greatest expected information value?

This census may supersede the currently nominated actor-call-60 builder-report
removal. That call is economically interesting, but another local redundancy
result may be less valuable than a checkpoint that tests the goal-to-probe
mechanism.

---

## Research practices that save cost without weakening claims

1. **Active case selection:** sample boundary cases and live rival explanations,
   not convenient calls.
2. **Successive halving:** reject candidates cheaply before longer evaluation.
3. **Matched checkpoints:** hold every request byte fixed except the declared
   treatment.
4. **Sequential replication:** start with two samples per arm; buy more only for
   genuine instability.
5. **Scripted positive controls:** establish that the measurement can observe the
   effect before interpreting a null.
6. **Physical oracle separation:** never expose successor or source-derived
   answers to the contestant context.
7. **Phase-specific builder budgets:** do not give terminal repair, goal work,
   and routine dynamics refresh identical authority or budgets.
8. **Proxy calibration:** use Gemma to screen Qwen changes only after paired
   packet evidence shows the relevant judgment transfers.
9. **Untouched holdouts:** use existing ar25 and LS20 traces for development and
   preserve new seeds/games for promotion.
10. **Stop at the claim ceiling:** do not buy a full game to answer a parser,
    artifact, immediate-dependence, or lifecycle question.

---

## Questions retired for now

- **Should we add more context?** Not until context is shown binding again.
- **Should we increase the host timeout?** Allocation caused the observed
  failure; a larger timeout would hide the exposure.
- **Should we improve planner search first?** Not before an objective and forward
  operators exist.
- **Should we tell the builder more forcefully to infer a goal?** Six explicit
  requests produced no predicate; another wording lacks a new mechanism.
- **Should we remove the builder globally?** Its value is phase-dependent and
  has not been isolated.
- **Should we run another broad Gemma game?** Not without a question that cheaper
  tiers cannot answer.
- **Did a tool help because it was called?** Help requires a changed decision and
  consequence, not presence or invocation count.

---

## Idea parking lot

Ideas here are not commitments. Promote one only when it answers a decision-linked
question more efficiently than the current strategy.

- Candidate-goal registry with supporting evidence, predicted terminal signature,
  cheapest discriminator, action deadline, and promotion state.
- Builder phase classifier and phase-specific cost/deadline policies.
- Actor-owned promotion authority with builder treated as a candidate-state
  producer.
- Counterexample curriculum generated from shortest branch-replay failures.
- Goal-probe score: whether the selected action distinguishes live objective
  hypotheses, with dual-purpose progress as a tiebreaker.
- Frozen cross-model packet arena for Gemma/Qwen proxy calibration.
- Scripted known-win lifecycle control for completion checkpointing.
- Scripted slow-provider control for action-return guarantees.
- Visualization of
  `observation → belief change → evidence → plan → action → consequence` with
  actor/builder ownership, time, cost, and uncertainty aligned by decision.
- Development/holdout registry preventing repeated adaptation to every available
  seed and game.

---

## Next consequential action

Build and run the **zero-model-cost epistemic-funnel miner** over the preserved
ar25 and LS20 checkpoint corpus. Use its earliest-failure distribution and
shortlist—not intuition or raw cost alone—to choose the next paid shadow.

The governing question is:

> Does evidence cross each boundary in time to change an action, and does that
> changed action improve the world?
