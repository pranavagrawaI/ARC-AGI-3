# Prospective factorized ARC-3 learner: candidate architecture and frozen KA59 test

## 0. Bind

| Item | Definition |
|---|---|
| Artifact | Executable-shaped architecture contract + frozen evaluation protocol |
| Objective | Test whether a small learned metacontroller can generate, preserve, test, and repair the hypotheses needed at KA59 forks without receiving the final KA59 rules |
| Anti-goal | Re-encode the final mechanics kernel and call it learning |
| Evidence | Prefix-only frames/actions/status; immutable raw trace; held-out suffix used by evaluator only |
| Falsifier | Learner omits needed alternatives, resolves unsupported rules, selects non-discriminating probes, leaks future information, or cannot beat simpler baselines under equal action budget |
| Move class | BUILD for prototype; COMMIT only after held-out-game replication and outside review |

**Researcher-hindsight qualification:** this protocol was designed after inspecting KA59. It can be frozen so the contestant is prefix-blind and can demonstrate online KA59 rediscovery, but it is not researcher-blind evidence of game-general learning. A genuinely hindsight-free architecture claim requires freezing the learner before revealing a protected third-party game.

## 1. Architecture boundary

### Deterministic substrate

The substrate may know **how to represent and test hypotheses**, but not KA59's rules.

```text
Perception primitives
  colors, connected components, candidate grid, geometry, frame differencing
  no labels: framed / solid / phase / F / target
  emit alternative segmentations/lattices when assignment is ambiguous

Object tracking
  candidate identity and geometry across frames
  ambiguity retained when assignment is not unique

Trace memory
  immutable (frame, action, coordinates, status, intermediate frames)
  provenance for every derived observation

Prediction commitment
  hash candidate identity, action prediction, and verification boundary before reveal

Candidate replay
  compile a candidate transition model
  replay prefix
  return first mismatch and affected factors

Search
  search only within a supplied candidate model
  no per-level scripts or game identifiers

Mismatch localization
  presentation-only vs causal-state vs terminal-status vs parser ambiguity
```

### Structured hypothesis state

```text
Entity hypothesis
  geometry        component / bbox / exact footprint / relation
  appearance      observed visual features only
  affordances     selectable / movable / passable / pushable / target-like / unknown
  latent_state    finite register / counter / orientation / none / unknown

Event hypothesis
  trigger         action / contact / phase wrap / occupancy / boundary / unknown
  scope           active / contacted / beam / all matching / local region / unknown
  operator        move / block / select / displace / transform / terminate / unknown
  parameters      direction / distance / blockers / continuation condition / period

Goal hypothesis
  relational predicate over entities, regions, occupancy, shape, and status

Belief record
  status          supported | contradicted | unresolved
  support         prefix observation IDs
  contradictions  prefix observation IDs
  predictions     probe → outcome
  code_length     frozen encoding cost
  dependencies    factors assumed
```

**Status semantics:**

- `unresolved`: compatible with prefix or not yet tested;
- `supported`: made at least one distinguishing correct prediction and has no known contradiction;
- `contradicted`: predicted an outcome excluded by observation.

`Supported` never means proven universally.

### Small learned metacontroller

The learned component may only:

```text
propose_factor_changes(prefix, belief_state)
propose_probes(state, live_hypotheses)
rank_probes(probe_matrix, costs, risks, decision_context)
choose_mode(inspect | replay | probe | plan | execute_prefix | repair | revert)
choose_verification_boundary(plan, live_hypotheses)
```

It does not:

- directly execute arbitrary hidden code;
- modify trace history;
- see held-out frames;
- receive final rule labels;
- provide an unverified plan directly to the environment.

## 2. Minimal hypothesis operations

### Candidate generation

At a mismatch or unexplained observation:

1. identify changed observed variables;
2. map them to factor slots;
3. propose **one-factor** alternatives first;
4. add interactions only if no one-factor model fits;
5. preserve prior compatible models;
6. compile each candidate and replay the complete prefix.

A repair proposed after a surprise may explain that observation, but receives no retroactive support from it. The repair remains unresolved until it makes a later, precommitted, distinguishing prediction.

Example without KA59 labels:

```text
Observation: contact moved the other component farther than predicted.
One-factor alternatives:
  distance parameter differs
  distance depends on terrain under victim
  motion continues until a visual predicate clears
  blocker set differs
  another event fired simultaneously
```

The learner is not told which is correct.

### Three-valued update

```text
for h in live_hypotheses:
  if observation excludes h.prediction:
      h.status = contradicted
  elif probe was distinguishing and observation matches h.prediction:
      h.status = supported
  else:
      h.status = unresolved
```

No “absence of evidence → false” transition exists.

### Model selection under observational equivalence

Replay is a hard admissibility gate, not a complete selector.

Among prefix-consistent candidates, rank by a frozen MDL/reuse objective:

```text
model_cost(h) =
    encoded factor count
  + encoded parameter count
  + exception count
  + duplicated operator cost
  + prefix prediction error penalty
```

Prefer lower cost, but retain any higher-cost model that predicts a different decision-relevant future outcome. The shared shove/wave operator can win by lower description length; it is never stamped “observed.”

## 3. Probe construction and ranking

### Probe matrix

For each candidate probe, deterministic simulation produces:

| Probe | Cost | Irreversibility | H1 outcome | H2 outcome | … | Plan relevance |
|---|---:|---:|---|---|---|---:|

A probe is invalid if all live hypotheses predict the same observable outcome.

### Ranking target

Train or tune only on non-test games. Freeze coefficients before KA59 evaluation.

```text
probe_value =
    decision_relevance
  × worst_case_partition_gain
  - action_cost
  - irreversibility_risk
  - budget_exhaustion_risk
  - expected_recovery_cost
```

Report every term. If outcome probabilities are not calibrated, use worst-case partition size rather than invented expected information gain.

### Verification cadence

Given a plan and live hypotheses:

1. precommit the selected model, next-action prediction, and verification boundary;
2. simulate the plan under each live model;
3. find the earliest action where any two predict different observable outcomes;
4. execute no farther than that boundary;
5. perform a cheap causal-state/status check after every action and interrupt immediately on mismatch;
6. inspect/full-replay at the boundary and after every theory edit;
7. if models agree, batch only through low-risk reversible states;
8. shorten further as remaining action budget shrinks.

This makes cadence a consequence of disagreement and risk, not a fixed interval.

## 4. Repair, revert, and promotion

```text
observe mismatch
  → localize channel (parser | causal | render | terminal)
  → propose smallest factor changes
  → compile candidates
  → replay full prefix
  → retain all prefix-consistent alternatives
  → rank by MDL/reuse
  → probe if alternatives affect current decision
  → promote chosen operational model
```

Promotion conditions:

- exact candidate identity fixed;
- prefix replay passed on causal state/status;
- presentation mismatch separately recorded;
- unsupported factors remain unresolved;
- no per-level script or future-derived constant entered the candidate;
- next verification boundary attached to the plan.

Revert when:

- the new observation contradicts the candidate;
- repair introduces historical regression;
- added complexity supplies no new fit or prospective distinction.

## 5. Frozen two-stage KA59 test

### Phase 1 — continuous prefix-blind shadow replay

Run through the complete 397-action admissible trace sequentially. Fork boundaries from [`../ka59-prospective-decision-audit.md`](../ka59-prospective-decision-audit.md) are **host-only evaluator indexes**; the learner never receives them.

At every step:

1. reveal only the current raw observation, legal actions, terminal state, and the learner's within-run memory;
2. require a hash-bound commitment to live hypotheses/statuses, predictions for legal actions, preferred probe/action, and verification boundary;
3. reveal the recorded action, then its successor observation;
4. allow one belief update;
5. persist the commitment and evidence provenance.

The evaluator alone holds suffix frames, fork annotations, final mechanics, and outcome-equivalence labels. This stage tests prospective reconstruction and prediction without changing the historical trajectory; it does not establish that the learner would choose a winning live route.

### Phase 2 — fresh live run

Run the frozen learner from KA59 level 1 with no archived action sequence or cross-run artifacts. Preserve only within-game memory across levels. Primary criteria:

- complete 7/7 with no GAME_OVER or reset;
- at most 397 charged actions;
- official RHAE 100%;
- no static level IDs, coordinate tables, exact suffix programs, or final-mechanics labels;
- runtime-generated hierarchical subgoals are allowed;
- no action after a detected causal mismatch without repair and replanning.

Phase-1 success without live success is retrospective prediction competence. Phase-2 success without precommitted fork evidence may be lucky policy behavior. The architecture claim requires both.

### Required learner response

Machine-readable at every transition commitment, with evaluator-scored summaries at host-only fork indexes:

```json
{
  "observations": [{"id": "...", "source": "frame/action/status", "claim": "..."}],
  "hypotheses": [{
    "id": "...",
    "factors": {},
    "status": "supported|contradicted|unresolved",
    "evidence": [],
    "assumptions": []
  }],
  "probes": [{
    "actions": [],
    "predictions_by_hypothesis": {},
    "cost": 0,
    "risk": {},
    "decision_relevance": "..."
  }],
  "selected_mode": "inspect|replay|probe|plan|execute_prefix|repair|revert",
  "selected_probe_or_plan": "...",
  "verification_after_action": 0,
  "operational_model": "...",
  "unresolved_not_relied_on": []
}
```

### Fork-level scoring

Score without revealing the final model to the learner:

1. **Alternative recall:** Did the live set contain an outcome class compatible with the next discriminating observation?
2. **Prediction separation:** Did proposed probe outcomes actually differ across live hypotheses?
3. **Probe efficiency:** Actions to a distinguishing observation.
4. **False resolution:** Unsupported hypothesis marked contradicted.
5. **Repair locality:** Number of factor slots changed after mismatch.
6. **Stale actions:** Actions after first observed/model disagreement before inspection.
7. **Prefix prediction:** Causal-state/status accuracy on the next held-out transition.
8. **Complexity:** MDL cost and exceptions.
9. **Decision relevance:** Did the probe resolve a factor needed by the current best plan?

The evaluator may retrospectively map learner hypotheses onto outcome equivalence classes. It must not require the learner to use the final model's names.

## 6. Baselines and ablations

### Baselines

| Baseline | Purpose |
|---|---|
| Leaked final KA59 kernel | Oracle upper bound; must be labeled invalid as learner |
| Single current model | Tests value of explicit alternatives |
| Text/prose memory only | Tests executable candidate/replay substrate |
| Random valid probe | Tests learned probe ranking |
| Cheapest separating probe | Deterministic metacontrol baseline |
| Fixed one-action verification | High-checking baseline |
| Full-route then verify | Low-checking baseline |
| Original EWMA trace choice | Historical reference, not gold policy |
| Original TWMA trace choice | Matched textual reference |

### Required ablations

1. remove three-valued unresolved state;
2. remove historical replay;
3. remove probe-outcome matrix;
4. remove MDL/reuse preference;
5. remove adaptive cadence;
6. remove factorized repair (allow whole-model rewrite);
7. remove trace provenance;
8. replace learned metacontroller with deterministic cheapest-separator.

A small learned model is only warranted if it beats the deterministic cheapest-separator on held-out prefixes/games, not merely the original traces.

## 7. Leakage controls

### Source isolation

- Run the evaluator and contestant in separate processes or hosts; mount only the current public observation channel.
- Do not mount final `world_model_*`, reports, reasoning logs, derived CSVs, unrevealed observations, or suffix frames.
- Hide archive path names and game/configuration identifiers.
- Start each evaluation arm in a clean process with frozen code/model hashes; preserve only that arm's explicit within-run memory.
- Log every file read and tool call.
- Block network, `/proc` leakage, inherited file descriptors, environment metadata, and cross-run workspaces.

### Memorization controls

KA59 alone cannot separate learning from memorization. Add:

- color permutations;
- object-ID permutations;
- spatial transforms that preserve mechanics;
- layout variants with the same factor structure;
- held-out games with different factor combinations;
- test prompts stripped of KA59-specific vocabulary.
- a host-secret semantics-preserving color/spatial isomorphism with corresponding action remapping, so pixel/coordinate scripts fail.

Transformations must be semantics-preserving and validated by the benchmark owner; otherwise they become a new task.

### Evaluator closure

- Hash prefix manifest and learner response.
- Verify no suffix/final artifact was accessed.
- Recompute objective action counts from metadata.
- Use original environment or an independently validated transition authority for probe outcomes.
- Keep pixel and causal-state scoring separate.

## 8. Acceptance ladder

### Stage A — format and sandbox check

The learner emits valid factorized hypotheses and three-valued status from prefixes. No claim of intelligence.

### Stage B — continuous frozen KA59 shadow replay

Accept only the claim:

> The learner can reproduce a specified fraction of KA59 branch decisions under prefix isolation.

Prerequisites:

- zero leakage violations;
- no dependence on final-rule names;
- every selected probe separates at least two live hypotheses;
- false-resolution rate reported;
- action/stale costs compared with deterministic baselines.

Do not change the pass set after seeing results; freeze all host-scored forks before running.

### Stage C — fresh KA59 live run

Require both the live criteria in Section 5 and the prospective commitments from Stage B. Passing only one limb is insufficient.

### Stage D — transformed KA59 variants

Accept transfer within the KA59 factor family only if performance survives validated visual/layout transformations without rule labels.

### Stage E — held-out ARC-3 games

Competition-agent evidence requires held-out games. Compare:

- official RHAE;
- completion;
- held-out transition prediction;
- stale actions;
- failed attempts/resets;
- compute/tokens/latency;
- hypothesis calibration.

### Stage F — deployment claim

Seek outside judgment before promoting defaults or claiming a general competition architecture. Report failure modes and games where the factor vocabulary is insufficient.

**Minimality rule:** “smallest” means local deletion-minimal, not globally minimal. If the full system and an ablation both pass the primary criteria under equal budgets, the removed component has not earned inclusion. In particular, if the deterministic cheapest-separator matches the learned metacontroller, delete the learned component from the claimed minimal architecture.

## 9. Failure interpretation

| Failure | What it means |
|---|---|
| Needed outcome class never proposed | Hypothesis generator/factor language insufficient |
| Correct alternative proposed but poor probe chosen | Metacontroller/ranking failure |
| Probe predicted no separation | Candidate simulator or probe constructor failure |
| Long stale suffix | Cadence/risk policy failure |
| Correct prefix fit, poor suffix prediction | Overfit/MDL selector failure |
| Simpler deterministic baseline wins | Learned metacontroller not justified |
| KA59 succeeds, transformed variants fail | Memorization/visual overfit |
| Transforms succeed, held-out games fail | Factor vocabulary too narrow |
| Pixel mismatch only | Rendering/perception issue, not necessarily mechanics |
| Full final kernel succeeds but learner fails | Discovery process, not mechanics capacity, is bottleneck |

## 10. Smallest credible empirical claim

If the first experiment succeeds, the justified claim is not:

> “We built a compact ARC-3 learner.”

It is:

> “Under prefix isolation, a factorized hypothesis manager plus a small metacontroller generated decision-relevant alternatives and selected lower-cost discriminating probes for frozen KA59 forks better than specified baselines.”

The bridge to a competition agent requires transformed and held-out games, fresh closed-loop execution, and official RHAE.
