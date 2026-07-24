# KA59 hypothesis-and-decision audit

**Run.** `ewma_v1.2_sv_gpt-5.5_xhigh_run1` / `ka59-38d34dbb`

**Sources.** Objective `step_index` trajectory; [`data/ka59-actions.csv`](./data/ka59-actions.csv) evidentiary outcomes; `agent.log` turn ranges; final executable artifacts; case study tables. Agent prose is supporting only.

**Claim scope.** Observed decision structure of this trajectory. Not a causal harness estimate. Not generalization beyond the recorded actions. The compact transition kernel below is an oracle-equivalent **final KA59 model**, not yet an ARC-3 learner that could discover that model from pixels and unknown actions.

## Decision map

| Item | Content |
|---|---|
| Artifact | Compact set of causal decision points + minimal architecture derived from successful choices |
| Objective | At each meaningful divergence: live competing hypotheses, available evidence, discriminating action/computation, choice made, and the minimal mechanism that could reproduce the successful choice |
| Evidence base | 397 actions; 23 turns; 285 verifiers; final 390/390 frame replay + 7/7 terminals; intent partition 195/104/52/32/14 |
| Verification | Every decision point cites global step `gN` and/or evidentiary outcome already present in [`data/ka59-actions.csv`](./data/ka59-actions.csv) or the case study |

Notation:

- **H** = live hypothesis at the fork
- **gN** = global action
- **O** = offline computation (no charged action)
- **Min** = minimal mechanism sufficient to lock the successful branch

## Two machines that must not be conflated

1. **The final KA59 mechanics model** already knows framed/solid/phase appearances, `F`, phase, waves, and matching targets. It explains the observed trajectory compactly.
2. **A prospective ARC-3 learner** starts with unknown patterns, actions, affordances, latent state, and goal. It must construct and manage alternatives before those concepts are known.

The reusable architecture is therefore not the KA59 transition function by itself. It is a factorized hypothesis language plus metacontrol:

```text
Entity: geometry | appearance | affordances | latent state
Event:  trigger | scope | operator | parameters
Goal:   relational predicate
Status: supported | contradicted | unresolved
```

The current trace identifies what that learner would need to represent. It does not yet show that a small learner could generate the right alternatives or choose the right probes online.

---

## I. Causal decision points

Each row is a fork where more than one model was live and a choice locked later behavior. Chronology is not the organizing principle; mechanism type is.

### D1. Terminal predicate

| Field | Content |
|---|---|
| Competing H | **H1** `5`/marked tile is a finish cell the actor enters. **H2** objects are movable pieces. **H3** completion = every outlined target covered by a distinct matching footprint. |
| Evidence before choice | Initial frame: actor-like `0`, marked `5`, hollow `4` outlines. One-step replay under H1 looked clean. |
| Discriminators | **g3** shove of `5` through field → ¬H1. **g4** selection of moved piece → objects are selectable entities. **g40/g42** entry into outline → outlines are passable targets. **g44** WIN only when both matching coverings hold → H3. Boundary probes **g13/g26/g27** → board edges are not terminals. |
| Computation | Weak one-step replay under H1 (passed). Later multi-piece placement + terminal status check. |
| Choice | Reject H1; keep selectable pieces; lock **H3** (matching footprint coverage). |
| Min | `done(s) ⇔ ∃ bijection targets↔entities with equal shape and position`. No finish-tile special case. |

### D2. Field passability under voluntary motion

| Field | Content |
|---|---|
| Competing H | **H1** `F` is free space. **H2** `F` blocks voluntary entry. **H3** `F` is always impassable. |
| Evidence before choice | Contact with `F` region; shove already crossed `F` under force (post-g3). |
| Discriminators | **g6** voluntary move into `F` blocked while forced path had crossed → H2, ¬H1, ¬H3. |
| Computation | Single discriminating directional action after contact setup. |
| Choice | Voluntary motion blocked by `F`; forced motion may cross. |
| Min | Two operators: `vol_move` refuses `F`/hard/occupied; `forced_move` may traverse `F`. |

### D3. Geometry representation

| Field | Content |
|---|---|
| Competing H | **H1** sprite/value identity. **H2** axis-aligned bbox only. **H3** normalized footprint `shape@position` (+ grid origin). |
| Evidence before choice | L1 1×1 success under weaker reps; L2 rectangles and outlined targets; L3 shifted origin and plus shapes. |
| Discriminators | L2 planned route under rectangular footprints succeeded after guards (**g45–93**). L3 full plan with zero exploration branch (**g94–128**) under arbitrary footprints + origin inference. |
| Computation | Parser rewrite; offline plan simulation; bounded real execution. |
| Choice | Lock **H3**. Drop sprite-primary identity for dynamics/goals. |
| Min | Entity = `(appearance, value, pos, shape[, phase_dir])`; target = `(shape, pos)`; match on shape+pos. |

### D4. Visual clock vs causal state

| Field | Content |
|---|---|
| Competing H | **H1** bottom-row pixels are causal state needing entity/direction branches. **H2** presentation-only spent-action clock. |
| Evidence before choice | Exact-frame verifier; repeated one-pixel / bottom-row mismatches with correct piece motion. |
| Discriminators | L2 guard **g48** clock-only stop → repair display, not motion. L2 **g79** contact-render stop → same split. L7 first-step one-pixel clock mismatch → budget fix, not dynamics edit. |
| Computation | Per-action or segment replay with localized mismatch images. |
| Choice | Treat clock as render budget; never as transition input. |
| Min | State transition on discrete board; renderer reads `steps_spent` separately. Causal equality ≠ full RGBA equality. |

### D5. Shove length / field continuation

| Field | Content |
|---|---|
| Competing H | **H1** contact displaces exactly 5 tiles (fit L1–L3). **H2** 5-tile impulse then continue while footprint remains on `F`, stopped by hard/occupied. |
| Evidence before choice | L1–L3 shoves matched fixed-5 after simplification (turns 4–5). No prior case left footprint on `F` after impulse. |
| Discriminators | **g146** (L4 local 18): real plus landed farther than H1. Hidden batch continued **g147–165** (stale). Repair → corrected suffix **g166–189**. |
| Computation | 37-action batch under H1; full-batch replay localized first mismatch; replan from exact real state. |
| Choice | Lock **H2**. |
| Min | `shove(e,dir) = forced_slide(e,dir, impulse=5, continue_while_on_F=true, blockers=hard∪occupied)`. |

### D6. Selectability ontology

| Field | Content |
|---|---|
| Competing H | **H1** all pieces selectable. **H2** only framed pieces selectable; solids/phase movable but not clickable. **H3** phase patterns selectable. |
| Evidence before choice | L1 framed selection works. Solids pushed on L3 without selection. No prior solid click. Phase `C`/`D` appearance new on L5. |
| Discriminators | Turn 9 O-replay: all-selectable compatible with history but solids were untested. The original model removed it operationally; epistemically it remained **unresolved**, not contradicted. **g190** phase click no-op → ¬H3. **g340** solid click no-op → ¬H1; five already-batched downs **g341–345** stale on still-active framed piece; full revert of engine/planner/notes. |
| Computation | Historical compatibility check (necessary, not sufficient); bounded real click test. |
| Choice | Operationally avoid relying on all-selectability, but preserve solid selectability as **unresolved** until **g340**. The original turn-9 rejection was conservative and happened to be correct; historical compatibility alone did not justify contradiction. After g340, lock **H2**. |
| Min | `ACTION6` changes active iff entity has framed appearance at coords; movability ̸⇒ selectability. |

### D7. Phase component ontology

| Field | Content |
|---|---|
| Competing H | **H1** large pattern = destructible source. **H2** source + passive target-side indicator. **H3** ordinary entities with optional `phase_dir`; multiple emitters. **H4** target slot at first plausible coordinate. |
| Evidence before choice | New `C`/`D` pixels; phase advances on directionals; incomplete coverage of emitter roles. |
| Discriminators | **g198** contact shoves pattern through `F` (not destroy) → ¬H1. **g210** target-side `DCC` emits horizontal wave → ¬H2. **g211** `(15,9)` nonterminal → ¬H4; true slot `(16,9)`. **g194** wrap displaces active piece → phase is causal. |
| Computation | 9-action acquisition; 12-action failed route; 1-action false terminal; 3-action corrected suffix; later O-compression to unified entities (turns 14–16). |
| Choice | Lock **H3** + exact target geometry; delete source/indicator roles. |
| Min | Phase piece ⊆ entities; directional acts advance shared phase; wrap ⇒ emit wave along `phase_dir`. Completion uses exact target coords. |

### D8. Voluntary motion after forced placement on `F`

| Field | Content |
|---|---|
| Competing H | **H1** `F` never voluntarily enterable (from D2). **H2** once carried onto `F`, voluntary traversal allowed. **H3** `F` freely passable always (pre-L6 route). |
| Evidence before choice | D2; L6 needs cross-field progress; H3 produced completing 45-action sim. |
| Discriminators | **O before g215:** historical replay of H3 fails at L1 g6-equivalent → discard H3, **0 actions spent**. **g221–232** planned experiment: carry onto `F` then voluntary step → rejected by replay (**g232**). **g220** upward wave can carry across `F` under force. |
| Computation | Full historical replay before execution; controlled prospective segment. |
| Choice | Keep D2. Cross-field progress only via forced operators (shove/wave). |
| Min | No state flag “on_F ⇒ vol_ok”. Forced carry is the only cross-`F` channel. |

### D9. Wave landing / shared forced family

| Field | Content |
|---|---|
| Competing H | **H1** wave = special corridor sweep (success cases overgeneralized). **H2** wave uses same blocker-aware forced landing as shove. **H3** wave reach ≡ shove distance (turn 19). **H4** wave affects only active piece. **H5** wave affects every other piece in path. |
| Evidence before choice | Successful carries (**g220**, later waves); no wall-blocked landing yet; no non-active wave victim yet. |
| Discriminators | **g268** predicted landing behind hard wall; real piece stayed below field → ¬H1. Stale **g269–276**. Repair → H2. **O turn 19:** H3 rejected by replay, 0 actions. **g302–339** prospective setup: wave moves non-active solid → **H5**, ¬H4 (**g339**). |
| Computation | Segment plans + replay localization; staged L7 setup with clock-only repair. |
| Choice | Lock the observed constraints: walls block wave landing and waves can move non-active pieces. Prefer **H2 + H5** as the smallest reusable model. A shared shove/wave operator is an MDL-style compression choice—simpler, replay-consistent, and reusable—not a logically forced observation; separate fitted operators remain observationally possible. |
| Min | `wave(src) = for e ≠ src in beam: forced_slide(e, dir(src), …)` identical landing law as shove; blockers include hard terrain. |

### D10. Execution control: when to trust a batch

| Field | Content |
|---|---|
| Competing H | **H1** long planned batch then end-of-batch replay. **H2** per-action or short-segment replay guards. **H3** open-loop full route. |
| Evidence before choice | L2 vs L4 vs L7 cost structure under same engine family. |
| Discriminators | L2 H2 stops at **g48/58/79/80** → **0 stale**. L4 H1: diverge **g146**, **19 stale** (**g147–165**). L6 H1: diverge **g268**, **8 stale**. L7 solid branch: probe + **5 stale** but contained to 6. |
| Computation | Same planner; different verify frequency. |
| Choice | Successful pattern = simulate → bounded execute → replay first mismatch → repair/replan. Failures cluster on H1 for high-risk model edits. |
| Min | After any model edit that changes transitions, max batch = shortest discriminating prefix; stop on first replay mismatch; never treat post-mismatch suffix as evidence for the pre-batch plan. |

### D11. Planner generality vs instance script

| Field | Content |
|---|---|
| Competing H | **H1** one generic best-first over footprint state solves all levels. **H2** generic core + phase primitive mode. **H3** hierarchical/staged subgoals or exact scripts when search fails. |
| Evidence before choice | Shared planner solves L1/L4–L6 class after mechanics lock; L2/L3 retain exact programs; L7 generic search fails. |
| Discriminators | L3 success is mechanics transfer, not proof of H1 (exact `level_3_planner_1` retained). L7 requires `_level_7_staged_plan` / 91-step staged program after H5 locked and H-select failed. |
| Computation | Offline search; fallback to staged construction after real trajectory taught subgoals. |
| Choice | Operational mix: H2 for several levels; H3 for L2/L3/L7. |
| Min for *successful choices about mechanics* | Mechanics kernel does not require generic planner. Planner min is: search in the same transition fn used for replay; allow hierarchical subgoals; do not treat instance scripts as learned laws. |

### D12. Offline compression after evidence locks

| Field | Content |
|---|---|
| Competing H | Keep trajectory-specific branches vs compress to shared operators once discriminators landed. |
| Evidence before choice | Multiple special cases after L1–L6 patches. |
| Discriminators | Turn 2 clock compression (action-neutral). Turns 9–12 shove/planner cleanup. Turns 14–16 phase ontology collapse. Turns 19–21 unify forced family. Each gated by replay of *already collected* traces. |
| Computation | Edit + historical replay only. |
| Choice | Compress only when replay of all prior traces still holds. |
| Min | `commit(rule)` only if `replay(all_traces, rule)=ok`; compression is not new evidence. |

---

## II. Retrospective identifiability certificates

These are **hindsight minima**: the smallest observed evidence sets sufficient to distinguish alternatives once the final hypothesis vocabulary is known. They certify retrospective identifiability. They are not an online experiment-selection algorithm, because the actor may not yet know which alternatives to generate, which latent variable explains a no-op, or which probe will isolate it. Extra actions after a certificate are recovery or exploitation relative to this retrospective hypothesis set, not necessarily avoidable online.

| Mechanism locked | Minimal lock set | What was *not* required |
|---|---|---|
| ¬ finish-tile | **g3** alone | Boundary probes; full L1 solve |
| Voluntary `F` blocked | **g6** after contact | Later L6 experiments |
| Selectable framed pieces | **g4** | All-selectable theory |
| Target = outline coverage | **g40** + **g44** (entry + multi-cover WIN) | Edge-as-goal probes |
| Footprint geometry | L2 rectangle success + L3 zero-explore plan | L7 |
| Clock non-causal | Any one pure clock mismatch (**g48** or L7 step-1) | Dynamics edits |
| Field-continued shove | **g146** predicted vs real endpoint | The 19 stale actions |
| Phase not selectable / shoveable entity | **g190** + **g198** | Source-destruction story |
| Multiple emitters + exact target | **g210** + **g211** | 12-action failed route body except first contradiction |
| ¬ voluntary on-`F` | Historical reject of free-`F` (**O**) + **g232** | 45-action unexecuted route |
| Blocker-aware wave landing | **g268** | 8 stale suffix actions |
| Wave hits non-active | **g339** after setup **g302–338** | Solid-select branch |
| ¬ solid selectable | **g340** alone | **g341–345** stale downs |
| Shared forced family | Conjunction of **g146**, **g268**, **g339** | Separate shove/wave code paths |

**Cost not in any minimum but paid anyway:** 32 stale batch actions; most of L1’s 33-action gap over the 11-step post-hoc plan; L4’s 19 post-divergence actions; L6’s failed voluntary-on-`F` setup; L7’s 5 post-click downs.

**High-value zeros:** free-`F` route killed by historical replay before **g215**; all-selectable rejected on untested-support grounds before real clicks (turn 9); wave-reach≡shove rejected offline (turn 19).

---

## III. Two derived architectures

### III-A. Oracle-equivalent KA59 mechanics kernel

Not the final 1,218-line artifact dump. This is the smallest final model that implements every successful KA59 mechanics choice **once the needed concepts and rules are already supplied**. It does not explain discovery from unknown pixels.

#### A. State

```text
Entity    = {id, appearance ∈ {framed, solid, phase}, value, pos, shape, phase_dir?}
Target    = {shape, pos}
State     = {entities, targets, active_id, phase_clock, hard, fields F, steps_spent}
done(s)   = matching cover of all targets by distinct entities
```

#### B. Transition kernel (single source of truth)

```text
vol_move(active, dir):
  one tile if free of hard ∪ F ∪ occupied else noop
  then tick_phase()

forced_slide(entity, dir, impulse, continue_on_F):
  move up to impulse; then while continue_on_F and footprint∩F and next free of hard∪occupied: step
  # identical for shove contact and phase wave victims

ACTION6(coord):
  active := framed entity at coord else noop

tick_phase():
  phase_clock += 1
  on wrap: for each phase emitter e: wave(e)

wave(e):
  for each other entity in beam(e.phase_dir): forced_slide(...)
```

#### C. Perception

```text
parse(frame) -> State   # grid origin, connected footprints, outlines, phase marks
render(state) -> frame  # optional; clock from steps_spent only
```

Causal verify compares `State` (and terminal flag). Pixel verify is diagnostic only.

#### D. Control loop (decision policy)

```text
loop:
  h := current kernel
  preserve each alternative as supported / contradicted / unresolved
  if unresolved rule becomes decision-relevant:
     propose discriminators and predict each outcome under every live hypothesis
     run the lowest-cost, lowest-risk, decision-relevant discriminator
  plan := search(h) or hierarchical_subgoals(h)
  prefix := shortest_prefix(plan) that can falsify recent edit
  execute(prefix)
  if replay(trace, h) fails at k:
     localize; patch h with minimal rule change;
     drop plan[k+1:];                       # stale suffix is not evidence
     replan from real state
  commit h only if replay(all_traces, h)
```

#### E. Explicitly excluded from the minimum

| Excluded | Why |
|---|---|
| Finish-tile / BFS-to-marker | D1 killed |
| Fixed-5 shove without `continue_on_F` | D5 killed |
| Source/indicator phase ontology | D7 killed |
| Voluntary-on-`F` flag | D8 killed |
| Active-only waves | D9 killed |
| Solid selectability | D6 killed |
| Trajectory-specific clock branches | D4 killed |
| `_level_7_staged_plan` / L2/L3 exact scripts | Solve aids, not laws (D11) |
| `LEVEL_STEP_BUDGETS` as dynamics | Presentation (D4) |
| all-`D` board-half direction heuristic | Fit, not locked by discriminator |

#### F. What this architecture explains

| Observed success | Mechanism in min-arch |
|---|---|
| L3 zero-explore solve | Footprint parse + forced_slide + vol_move enough to plan |
| L4 repair durable into L6–L7 | `continue_on_F` inside forced_slide |
| L6 cross-field without free `F` | wave/shove forced_slide only |
| L7 solid through field | wave over all non-source entities |
| Contained L7 select error | short discriminator + revert + replan |
| 390/390 final replay | single kernel + parse consistent with trace |
| Stale-action tax | violation of prefix/replay rule (D10) |

#### G. What it does **not** claim

- That search without hierarchical help solves L7 from initial state.
- That parse heuristics generalize off this game family.
- That 282 post-hoc plan length is attainable in a fresh real run.
- That executable harness *caused* the textual-run gap (matched observation only).

### III-B. Prospective ARC-3 learner hypothesis

```text
Deterministic substrate
  perception primitives and object tracking
  immutable trace memory
  replay and mismatch localization
  transition search

Structured hypothesis state
  Entity  = geometry × appearance × affordance × latent_state
  Event   = trigger × scope × operator × parameters
  Goal    = relational predicate
  Belief  = supported | contradicted | unresolved

Small learned metacontroller
  propose one-factor repairs and alternative explanations
  propose probes and predictions under each live hypothesis
  rank probes by information value, action cost, downstream risk, and decision relevance
  choose inspect | replay | probe | plan | execute_prefix | repair | revert
  choose verification cadence and batch length

Model selector
  reject contradicted hypotheses
  retain unresolved alternatives
  among replay-equivalent models prefer lower description length, greater operator reuse,
  and better prospective predictions—without calling that preference an observation
```

This is the architecture claim worth testing. The mechanics kernel is only a target concept inside its structured hypothesis space.

---

## IV. Dependency graph of locks

```text
D1 done-by-coverage ──────────────┐
D2 vol vs forced on F ────────────┼─► D5 continue_on_F shove ─┐
D3 footprints ────────────────────┤                          ├─► D9 shared forced + all-piece wave
D6 framed-only select ────────────┤                          │
D4 clock ̸= causal ───────────────┤                          │
D7 phase as entities ─────────────┴─► D8 no voluntary-on-F ──┘
                                         │
D10 bounded execute/replay ──────────────┴─► cheap recovery (L2, L7) vs stale tax (L4, L6)
D11 planner hierarchy ──────────────────────► completion when kernel search insufficient
D12 replay-gated compression ───────────────► final small kernel from patched special cases
```

Critical path for the oracle-equivalent mechanics kernel:

```text
D1 → D2 → D3 → D5 → D7 → D8 → D9
```

with **D6** independent but required before L7 exploitation. **D10 is not another game rule:** it is the metacontrol policy that determines identification cost and is the strongest candidate for learned capacity.

---

## V. After

| Item | Status |
|---|---|
| Changed | New audit artifact: `ka59-hypothesis-decision-audit.md` |
| Checked | Decision points tied to CSV evidentiary outcomes and case-study global steps |
| Not checked | Fresh agent.log line re-read for every turn prose claim; held-out counterfactual kernel run |
| Learned | The final mechanics reduce to ~12 forks, but a learner must generate and maintain those alternatives prospectively; cost is concentrated in metacontrol |
| Remains risky | Oracle mechanics kernel not executed as a learner; prospective hypothesis generation and experiment selection remain untested; L7 still needs search/hierarchy beyond kernel |
| Overturn | Any held-out ka59/other-game trace where forced_slide+footprint+framed-select fails while the full final tree succeeds |
| Next test | Freeze min-arch without scripts/clock budgets; predict held-out suffix; measure causal-state match and real RHAE |

**Label.** Analysis synthesis from existing verified tables. Not a new extractor run. Not suite green.
