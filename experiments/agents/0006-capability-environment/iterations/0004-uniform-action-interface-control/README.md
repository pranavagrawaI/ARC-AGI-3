# Iteration 0004 — Uniform action-interface control

## Diagnosis

The frozen `0002-R` control exposed actions 1–5 only as parameterless integers while
action 6 alone carried `x`/`y` coordinates. Before observing any transition, the model
correctly inferred action 6 was a click and selected it. The old control used action 6 on
44/44 actions, so later learning-loop treatments did not originate the fixation.

## Corrected control contract

`uniform_action_control.corrected_control_spec()` derives a new control generation from
`0002-R` and adds only these public input-event names:

- 1: `UP`
- 2: `DOWN`
- 3: `LEFT`
- 4: `RIGHT`
- 5: `SPACE`
- 6: `MOUSE`, with grid coordinates `x` and `y`

The prompt explicitly says these names describe inputs, not their effects. Consequences,
object roles, rules, and objectives remain evidence-derived.

## Deliberate omissions

- no action/result demonstration;
- no JSON example beyond the pre-existing shared response schema;
- no exploration policy;
- no instruction to test every action;
- no game-specific semantics;
- no learning-loop or coaching capability;
- no mutation or re-pin of the old frozen control.

## Offline verification

```bash
uv run --with pytest pytest -q \
  experiments/agents/tests/test_p13_uniform_action_interface.py
uv run python3 \
  experiments/agents/0006-capability-environment/iterations/0004-uniform-action-interface-control/verify_offline.py
```

The first command verifies the model-visible contract and old-control freeze. The second
runs the corrected specification through the ordinary offline episode and artifact path;
it makes no model or network call.

A live run was separately authorized and completed. Its behavioral question was:

> With all six input events uniformly named, does action choice reopen before any further
> epistemic or coaching mechanism is added?

## Authorized live generation

`uniform_action_live.py` prepares and executes one matched `cd82` run. It fixes the
recorded control's model, seeds, budgets, context policy, and retry policy; pins the
recorded effective provider to `modelrun/fp4` with fallbacks disabled; commits every
participating source and the recorded control manifest; and forbids a control rerun,
treatment, or automatic successor generation.

```bash
uv run python3 experiments/agents/0006-capability-environment/uniform_action_live.py prepare
uv run python3 experiments/agents/0006-capability-environment/uniform_action_live.py \
  execute --confirm 'EXECUTE UNIFORM ACTION CONTROL CD82 V1'
```

## Live result

| Measure | Biased-interface control | Uniform-action control |
|---|---:|---:|
| Actions | 44 | 44 |
| Action 1 | 0 | 9 |
| Action 2 | 0 | 13 |
| Action 3 | 0 | 13 |
| Action 4 | 0 | 9 |
| Action 5 | 0 | 0 |
| Action 6 | 44 | 0 |
| Levels completed | 0 | 0 |
| Model calls | 60 | 52 |
| Reported cost | $0.24016772 | $0.20080473 |

The correction eliminated action 6's monopoly in this run. It did not produce broad
experiment coverage or progress. The model learned genuine directional transitions, then
closed around an unsupported movable-player/white-target objective and never executed
SPACE or MOUSE despite considering both late in the run. Post-run game-source inspection
showed that actions 5 and 6 were required to construct the target pattern.

This is one model-seed realization. It establishes a material interface effect, not a
population effect or a claim that the names-only interface is fully neutral. Reproducible
derived evidence is in `evidence.json`; complete raw evidence is under the ignored
`artifacts/live-control-cd82-v1/` directory.

This is not a byte-identical historical pair. The model-visible manifests differ only in
system-prompt hash and length, and the old freeze verifies without drift, but five shared
host source files changed and three P9 files were added between generations. The provider
was also tightened from historically unpinned routing that returned ModelRun to an explicit
`modelrun/fp4` pin. Treat the result as strong diagnostic evidence, not isolated causal
effect estimation.
