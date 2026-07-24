# KA59 matched prospective audit: executable vs textual GPT-5.5 xhigh

## Scope and authority

| Item | Executable | Textual |
|---|---:|---:|
| Configuration | `ewma_v1.2_sv_gpt-5.5_xhigh_run1` | `twma_v1.2_gpt-5.5_xhigh_run1` |
| Official game RHAE | 100.000% | 43.3546876334% |
| Charged actions | 397 | 1,687 |
| Attempts by level | `[1,1,1,1,1,1,1]` | `[1,1,2,1,1,2,6]` |
| Resets | 0 | 7 |
| Final outcome | WIN, 7/7 | WIN, 7/7 |

**Evidence authority:** action and attempt counts come from objective metadata deduplicated by global `step_index`. Textual mechanics descriptions come from its level reports/reasoning logs and are treated as trace-supported agent claims, not an independently replayed executable model. Executable mechanics have the additional 390/390 + 7/7 replay evidence described in the case study.

**Design limit:** one matched run per harness is descriptive, not causal. “Attributed” below means action-accounting or mechanism-consistent association, never an identified treatment effect.

**Headline reconciliation:** the user-supplied `1,388` textual headline is incompatible with the authoritative runner total and supplied per-level vector, which both give `1,687`. Retaining `1,388` would require an undocumented exclusion of **299 charged actions**. No trace-backed accounting category supports that exclusion, so `1,388` cannot be combined with the per-level vector or RHAE reconstruction.

## 1. Objective action accounting (no causal interpretation)

### Per level

| Level | EWMA actions | TWMA actions | TWMA − EWMA | TWMA failed attempts | Uncapped weighted RHAE delta, EWMA−TWMA (points) |
|---:|---:|---:|---:|---:|---:|
| 1 | 44 | 12 | -32 | 0 | -2.660862 |
| 2 | 49 | 47 | -2 | 0 | 0.000000 |
| 3 | 35 | 171 | +136 | 1 | +11.368388 |
| 4 | 61 | 50 | -11 | 0 | -4.877062 |
| 5 | 25 | 50 | +25 | 0 | +12.757143 |
| 6 | 87 | 268 | +181 | 1 | +19.444436 |
| 7 | 96 | 1,089 | +993 | 5 | +26.509632 |
| **Total / sum** | **397** | **1,687** | **+1,290** | **7** | **+62.541674 uncapped** |

The executable run's uncapped weighted score is 105.896362%, then the environment cap subtracts 5.896362 points. Therefore:

```text
62.541674 uncapped weighted difference
−5.896362 executable environment-cap adjustment
=56.645312 official RHAE-point difference
```

The score gap cannot be allocated linearly from raw action differences because RHAE squares `H/A`, weights later levels more, caps each level at 115%, and caps a completed environment at 100%.

### Failed-attempt/recovery partition

Use one uniform convention: every reset is recovery cost, including the reset that starts the eventual successful attempt.

| Level | Failed-attempt moves | Resets | Recovery subtotal | TWMA successful-attempt moves | EWMA moves | Successful-route difference |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 12 | 44 | -32 |
| 2 | 0 | 0 | 0 | 47 | 49 | -2 |
| 3 | 100 | 1 | 101 | 70 | 35 | +35 |
| 4 | 0 | 0 | 0 | 50 | 61 | -11 |
| 5 | 0 | 0 | 0 | 50 | 25 | +25 |
| 6 | 150 | 1 | 151 | 117 | 87 | +30 |
| 7 | 1,000 | 5 | 1,005 | 84 | 96 | -12 |
| **Total** | **1,250** | **7** | **1,257** | **430** | **397** | **+33** |

Thus:

```text
1,257 failed-attempt + reset actions
   33 net difference among successful-attempt routes
=1,290 total action gap
```

This is the strongest non-causal decomposition. It says the gap is primarily **failure avoidance**, not that EWMA's eventual successful route was uniformly shorter. In level 7, TWMA's sixth-attempt route used 84 moves—12 fewer than EWMA's 96—but needed 1,005 prior failed/restart actions to discover it.

Failed attempts plus resets account for **1,257 / 1,290 = 97.44%** of the raw gap. This is an accounting fact, not evidence that executable verification caused those failures to be avoided.

## 2. Prospective alignment by causal problem

| Causal problem / equivalent state | Prefix-supported options in both runs | EWMA behavior | TWMA behavior | Mechanism-consistent difference | Action evidence | What cannot be claimed |
|---|---|---|---|---|---:|---|
| **Learn basic objective and selection (L1)** | Marked tile as goal vs movable piece; coordinate action select vs no-op; one vs all outlines | Built executable parser/renderer/planner but explored broadly; 44 actions | Direct visual inspection and targeted actions; solved in 12 | Textual manual reasoning was cheaper for initial acquisition; executable construction did not itself save actions | TWMA − EWMA = **−32** | Executability is not uniformly beneficial; no later benefit can retroactively erase L1 cost |
| **Shapes + `F` transfer (L2)** | Active crossing vs forced inactive crossing; shape-to-socket mapping | Footprint model + replay guards; 49 | Manual geometry; one false active-2×2 crossing; 47 | Near tie; both found same high-level mechanics | **−2** | No evidence of an EWMA advantage here despite more formal verification |
| **Plus geometry and simultaneous targets (L3)** | Filled bbox vs notched footprint; one plus can service two targets vs simultaneous occupancy; wall bounce vs direct notch push | Parsed exact plus footprint and executed complete 35-action plan with no reported exploration branch | Attempt 1 spent 100 discovering all targets and moving one plus; attempt 2 rejected bounce and recognized notches via local simulation, then solved in 70 | Better geometric representation + planning and much shorter repair latency are consistent with avoiding attempt failure | **+136** = 101 recovery +35 successful-route | Cannot isolate parser, search, prior transfer, or harness; TWMA did eventually construct the correct notch model and preserved it |
| **Field-continued shove (L4)** | Fixed 15-pixel/five-tile push vs terrain-conditioned continuation | Long 37-action batch under wrong model; g146 divergence +19 stale; 24-action repair; total 61 | One transfer landed farther than expected; one extra blocked push; switched to receiving-side continuation; total 50 | Textual inspection repaired faster in action terms; EWMA's batching erased replay's potential benefit | **−11** | Exact replay did not guarantee lower cost; verification cadence mattered more than verifier existence |
| **Phase/state machine (L5)** | Colored blocks as objects, displays, triggers, emitters; goal object vs clear socket | Controlled 9-action acquisition, failed 12-action route, 1-action target correction, 3-action solve; 25 | Mapped six-phase register through movement probes; cleared central block and right socket; 50 | Executable latent-state representation + simulation/primitive planning are consistent with lower exploration | **+25** | Cannot assign each saved action to replay vs planner vs different probe path |
| **Cross-field phase transfer (L6)** | Voluntary `F`, reset/wave transfer, block/elevator positions, wall constraints | Historical replay killed one free-`F` route at zero actions; prospective probes + repaired wave model; one attempt, 87 | Attempt 1 spent 150 finding slider/elevator and return transfer; reset; attempt 2 followed learned route with detour, 117 | Offline rejection, explicit state transition model, and planning are all consistent with avoiding a full failed attempt | **+181** =151 recovery +30 route | The final descriptions differ (“waves” vs “reset elevator”); matched state/action equivalence must be established before treating terminology as the same mechanism |
| **Move plus across L7 barrier** | Reuse phase transfer; inactive-wave scope; coordinate select; hidden route around barrier; edge as wall vs open | Proposed non-active wave scope; 38-action prospective setup succeeded; solid click failed in bounded six-action branch; corrected route; one attempt, 96 | Five 200-action attempts tested many reset lanes, coordinate actions, shape gates, and lower-block analogies; attempt 6 tested open left edge, moved horizontal off-screen, and solved in 84 | Hypothesis generation and metacontrol dominate: EWMA found a viable mechanism early; TWMA retained memory but searched an analogy-heavy family for five budgets before testing boundary openness | **+993** =1,005 recovery −12 eventual-route advantage | EWMA and TWMA used different successful mechanisms; one cannot say executable replay merely found the same route faster |

## 3. Requested mechanism audit

### A. Better hypothesis

**Strongest evidence: L3 and L7.**

- L3 EWMA's exact footprint representation made plus notches available to planning before acting. TWMA's attempt 1 behaved as though the plus were a filled block; attempt 2's local simulation exposed the notch rule (`level_3_reasoning_log.md:11`).
- L7 EWMA proposed that a phase wave could affect a non-active solid and tested it. TWMA instead pursued level-6-style reset-transfer analogies for five attempts, then found a different open-boundary route (`level_7_reasoning_log.md:61–64`).

**Bound:** these are observed hypothesis-path differences, not proof the harness caused them.

### B. Preserved memory

The simple claim “TWMA forgot; EWMA remembered” is **not supported**.

TWMA's world model and level-7 reasoning log preserve an extensive negative-test ledger across six attempts (`world_model.md:32–57`; `level_7_reasoning_log.md:11–60`). It remembered:

- failed reset alignments;
- coordinate no-ops;
- shape-gate constraints;
- lower-block immobility;
- simultaneous target requirements.

The difference is better described as **structured, replay-checked executable memory versus persistent prose memory**—a distinction in representation and use, not memory versus no memory:

- EWMA could replay candidate transition rules mechanically and search over executable state.
- TWMA retained prose memory but still had to reason manually about coverage and untested regions.

### C. Rejected a bad rule offline

**Concrete positive case:** before EWMA g215, a 45-action plan requiring voluntary `F` passability was rejected by historical replay at zero environment cost.

TWMA did inspect prior reports and reuse mechanics, but it lacked a mechanically checked transition model. In L7 it paid real actions to test many reset-transfer variants. These are not exact paired hypotheses, so no exact action saving can be assigned to replay alone.

### D. Shortened repair latency

**Mixed.**

- EWMA replay localized L2 disagreements without stale suffixes and localized L4/L6/L7 mismatches precisely.
- But EWMA's L4 long batch incurred 19 stale actions, while TWMA repaired its analogous long-transfer misconception after one wasted blocked push and solved in 50.
- TWMA's major latency failures appeared at attempt scale: 100 actions in L3, 150 in L6, and 1,000 in five L7 attempts before a successful hypothesis family.

Conclusion: executable mismatch localization can shorten diagnosis, but only if verification cadence checks before the action budget or batch has already been spent.

### E. Enabled better planning

**Consistent evidence:**

- EWMA L3: 35-action complete plan from the initial state.
- TWMA L3: attempt failure, then 70-action successful attempt.
- EWMA L5/L6 also used explicit state transitions and search; TWMA used local simulation plus hand-authored sequences.

**Limit:** EWMA's L7 “generic” planner failed and gave way to a staged instance-specific program. TWMA's eventual off-screen route was shorter than EWMA's real route. So the advantage is not “generic planner always better”; it is the ability to search a reliable transition model when the right state factors are present.

### F. Merely changed batching

Both runs used batches. TWMA's log includes short loops and longer stop-checked sequences; EWMA issued 341 of 397 actions through scripts.

The important distinction is not batch vs no batch. It is:

```text
Does the batch stop at the first state where live models predict different outcomes?
```

EWMA measured 32 stale actions after hidden mismatch. TWMA has no equivalent executable mismatch boundary, so a directly comparable stale-action count is not yet available. Its dominant cost is complete failed attempts, not proven intra-batch staleness.

## 4. Non-overlapping attribution: what can be quantified

Only the following partition is mechanically non-overlapping:

| Bucket | Actions |
|---|---:|
| TWMA failed attempts | 1,250 |
| TWMA resets | 7 |
| Net successful-route difference | 33 |
| **Total TWMA excess** | **1,290** |

A causal partition such as:

```text
X replay + Y memory + Z planning + ...
```

is not identified because those mechanisms overlap. For example, L3's avoided failure could be described simultaneously as better perception, better hypothesis, better planning, executable replay, or shorter repair latency. Assigning it to one bucket would be arbitrary.

The defensible mechanism map is therefore many-to-many:

| Observed gap | Compatible mechanisms |
|---|---|
| L3 failed attempt | geometry hypothesis, state representation, planning, repair latency |
| L5 25-action gap | latent-state representation, probe efficiency, primitive planning |
| L6 failed attempt | offline rejection, phase-state persistence, planning, repair latency |
| L7 five failed attempts | hypothesis generation, analogy fixation, experiment ranking, verification cadence, budget awareness |
| EWMA L1/L4 disadvantage | construction/exploration overhead; long-batch metacontrol failure |

## 5. Prospective lesson from the matched pair

The pair does not support “make the world model executable” as a sufficient policy. It supports a narrower architecture hypothesis:

```text
Trace memory alone is insufficient.
Executable replay alone is insufficient.
Search alone is insufficient.

The useful combination is:
  factorized state
  + live unresolved alternatives
  + cheap historical rejection
  + probe ranking
  + execution only to first disagreement
  + mismatch localization
  + replan from observed state
```

The RHAE gap is driven mainly by whether the system avoids exhausting whole attempts before discovering that its hypothesis family is wrong. That is a metacontrol claim, not merely a mechanics-model claim.

## 6. Next discriminating comparison

Freeze equivalent prefixes at:

1. L3 before the first plus interaction;
2. L6 before any phase-transfer action;
3. L7 initial state and after both domino routes are known but plus access is unresolved.

Give both candidate learners identical prefix observations and equal action budgets. Require them to return:

- live hypotheses with supported/contradicted/unresolved status;
- candidate probes and outcome matrix;
- selected probe and rationale;
- verification checkpoint;
- subsequent plan.

This tests the proposed experiment policy without confounding it with different prior trajectories or post-hoc final models.
