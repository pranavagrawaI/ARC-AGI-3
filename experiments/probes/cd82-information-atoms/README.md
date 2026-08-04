# cd82 information-atom probe

This probe asks which minimum prefix-visible information changes Gemma's response after a contradicted directional prediction.

## Frozen branch

- Corrected-control run: `live-control-cd82-v1`
- Prior decision: process event 71, action 3, predicted LEFT movement toward the white square.
- Public result: process event 72, zero changed pixels and identical before/after digests.
- Replayed request: process event 73, `request-payloads/000073.json`.
- Provider: `modelrun/fp4`, fallbacks disabled.

The archived request already contains the public result. Conditions test retrieval and relation-making rather than adding source truth.

## Conditions

| ID | Added information |
|---|---|
| C0 | None; byte-identical request |
| C1 | Retrieve the prior prospective belief |
| C2 | Link the belief to the already-visible result |
| C3 | Label the pair as contradiction of a state-independent rule |

Each treatment adds exactly one top-level `information_probe` field to the current environment-observation JSON. No game environment is created or stepped.

## Offline verification

```bash
uv run python3 experiments/probes/cd82-information-atoms/verify_offline.py
```

## Prepare

Preparation performs endpoint discovery and writes a deterministic, source-hash-bound plan. It makes no model call.

```bash
uv run python3 experiments/probes/cd82-information-atoms/probe.py prepare
```

## Execute

Execution requires both the exact phrase and exact prepared plan hash. Four model calls are the hard maximum; environment actions are structurally zero.

```bash
uv run python3 experiments/probes/cd82-information-atoms/probe.py execute \
  --confirm 'EXECUTE CD82 INFORMATION ATOMS V1' \
  --confirm-plan-sha256 '<prepared sha256>'
```

A broad instruction to build or prepare this probe does not authorize these calls. The exact frozen plan must be reviewed first.

## Review

`build_evidence.py` extracts selections and integrity evidence. It leaves semantic fields unset until a response-hash-bound `semantic-review.json` exists; it then merges that human-authored review. Apply `review-rubric.json` to all complete response bodies before deciding whether to kill, revise, confirm live, or replicate a winner.

## Result

The authorized bundle completed on the frozen `modelrun/fp4` route:

- 4 model calls;
- 0 environment actions;
- all request and response hashes verified;
- total reported cost: `$0.0143515`;
- C0 and C3 selected Python wall inspection;
- C1 and C2 selected action 1 / UP;
- all four retained the movement-plus-wall theory;
- none withdrew the global control rule, proposed conditional control semantics, or selected a discriminator between control hypotheses.

An independent blind reviewer agreed that every response retained blockage rationalization and none withdrew the global rule. It classified wall hypotheses and wall-opening probes as generic conditional/discriminating behavior, then called all four responses potent. That cannot identify an intervention effect because one response was C0, every response saw the same underlying mismatch, and no treatment changed the governing theory. `semantic-review.json` preserves both the positive review and this attribution objection. Generic obstacle experiments do not satisfy the narrower construct of conditional **control** semantics and discrimination between **action-semantic** alternatives.

Verdict: `kill_mismatch_information_at_zero_change_branch`.

This is branch-specific. A zero-change transition is compatible with ordinary obstacle dynamics, so contradiction linkage at this branch cannot distinguish hidden controls from a wall. The licensed next probe is a compact public object-feature difference at a large non-rigid transition—extent, area, centroid, and shape signature—without naming the transformation or supplying game semantics.

Reviewed evidence is in `evidence.json` and `semantic-review.json`. Raw requests and responses remain ignored under `artifacts/bundle-v1/`.
