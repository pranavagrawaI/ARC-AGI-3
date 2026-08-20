# tycho-nooa-no-wm-ar25-20260819T173935Z

Question: with the world-model machinery removed, does the agent do the one
thing five runs of machinery never did — choose an action to discriminate
between its own goal hypotheses, or write down a checkable success criterion?
Posture: causal probe on authority topology, not gameplay. First of Tycho's
four modes to be run other than `trigger`.

## Why this arm, and why first

Every run to date is `trigger`. The standing finding is that the physics model
reached `simulation_accuracy=1.00` and produced 0 levels — an *exact* model the
loop never converts into a goal test — so the question is no longer "is the
model good enough" but "is the modelling apparatus displacing the reasoning
that would identify the task".

`no_world_model` is the only arm with independent prior evidence. Probe B
(`experiments/cards/probe-b-correspondence-20260819T015200Z/`) handed the same
model (`meta/muse-glimmer-30b`) the opening frame one-shot, with no machinery,
and it named the correct body **8/8** — with the mirror argument and the 40-vs-45
cell count. The variable Probe B removed is exactly the variable this arm
removes. It is also the cheapest arm to build (upstream already has every
switch) and the cheapest to run: the builder was 165 of 236 calls and $1.122 of
$1.887 on 162040Z.

Contrast: `tycho-nooa-level-start-ar25-20260818T175557Z` — same game, same seed,
same 60-action cap, `trigger` mode, 0 levels, goal hypotheses enumerated at pass
3 and restated unchanged at pass 34. That is a same-budget control already in
hand, which is why the cap is 60 and not 20.

## Measures

Ranked by what they can discriminate. Measure 1 is descriptive only: memory
records that 175557Z formed goal hypotheses at pass 3, so it will fire in every
arm and cannot separate them.

1. (descriptive) turn of the first explicit candidate objective.
2. **turn of the first action explicitly chosen to distinguish competing
   objective hypotheses**, named as such in the reasoning trace. Exactly zero
   across every run so far.
3. **turn of the first written, checkable success criterion** — a predicate the
   agent could evaluate against a frame, in `notes/actor_beliefs.md` or in a
   `run_python` call. Re-specified from "encoded in `outcome()`" because this arm
   has no `world_model.py`; the substance is a *prospective* target, not the
   channel it is written in.
4. **does the agent ever move or name Shape B** — the body congruent to the
   target — as the object that belongs there. The sharpest game-specific form of
   measure 2, and the thing Probe B says this model can answer one-shot.
5. levels completed.

Secondary: cost and call count against 162040Z's actor share ($0.764 / 71 calls
over 20 actions), to price the remaining arms.

Prediction: measure 1 fires by turn 3 as in every prior run. Measure 2 is more
likely here than in `trigger` but I expect it absent — my point estimate is
absent, and I would not be surprised to be wrong. Measure 4 present is the
result I would bet against at even odds and most want to see. Levels: 0.

Cap: ar25 seed 0; 60 actions; stop-after-levels 1; one live model; 120 min; no
retries. `TYCHO_MODE=no_world_model`, `TYCHO_NOOA_CAPTURE_REASONING=1`.
Stop: action cap, 1 level, terminal, wall-clock, transport failure.
Invalid if: crash before action 5; `world_model.py`, `verify.py` or `plan.py`
present in the workspace (the arm did not take); no reasoning text in
`llm-trace.jsonl` (this run is read from reasoning, not from tool traces).

Result map:
- measure 2 or 3 fires -> the modelling apparatus was displacing task reasoning.
  That is a large, cheap, decision-changing result and it makes `orchestrator`
  (the expensive arm) worth building next.
- both null at 60 actions with the same-budget `trigger` control also null ->
  the topology is not the operative variable; goal acquisition needs an explicit
  strategy and controller state, not a different authority arrangement. Go to
  the NOOA `_acquire_goal` strategy rather than to `single`/`orchestrator`.
- measure 4 present but 2 and 3 null -> the agent can see the correspondence and
  still cannot turn it into a test. That isolates the defect to *conversion*, not
  perception, and is the strongest argument for controller state
  (`objective_status: unbound`) over any further prompt.

Note: n=1 per arm cannot separate a topology effect from sampling in a
stochastic loop. Only a large effect counts here — measures 2/3/4 going from
exactly zero to non-zero. A shift in degree is not a result at this n. See
`wm-lane-uptake-and-sandbox-lifetime` for a single-run effect that did not
replicate.

Code: 738c94f + uncommitted mode plumbing (`tycho_nooa.modes`, imported from
upstream's `tycho.agent.modes`), `TYCHO_MODE` on arc-host's worker allowlist.
65 solver tests + 14 arc-host tests green, ruff clean.
