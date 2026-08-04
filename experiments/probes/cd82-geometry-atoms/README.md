# cd82 geometry information atoms

This zero-action shadow probe tests whether compact public component geometry reopens action semantics after Gemma interpreted a large transition as ordinary movement.

## Frozen branch

- Source run: `live-control-cd82-v1`
- Prior decision: event 79, LEFT to move a diamond-shaped purple player toward a white target.
- Public transition: event 80 / step 15, 201 changed pixels.
- Replayed request: event 81, `request-payloads/000081.json`.
- Original next decision: DOWN toward the target.

## Deterministic public measurement

Largest 4-connected component of color ID 15:

| Feature | Before | After |
|---|---:|---:|
| Extent | 13×13 | 12×7 |
| Area | 85 | 84 |
| Centroid | [40.764706, 29.235294] | [31.5, 28.0] |
| Normalized shape | changed | changed |

No object identity, rotation, control rule, objective, game-source fact, or future observation is supplied.

## Conditions

- C0: exact archived request.
- C1: component extent only.
- C2: extent, bbox, area, centroid, normalized-shape hashes.
- C3: C2 plus the geometric fact that rigid translation preserves extent and normalized shape.

Every treatment changes only `messages[16].content[0].text.information_probe`.

## Offline verification

```bash
uv run python3 experiments/probes/cd82-geometry-atoms/verify_offline.py
```

## Prepare

Endpoint discovery only; no model call:

```bash
uv run python3 experiments/probes/cd82-geometry-atoms/probe.py prepare
```

## Execute

Requires exact phrase and exact prepared plan hash. Hard maximum: four model calls, zero environment actions.

```bash
uv run python3 experiments/probes/cd82-geometry-atoms/probe.py execute \
  --confirm 'EXECUTE CD82 GEOMETRY ATOMS V1' \
  --confirm-plan-sha256 '<prepared sha256>'
```

Broad permission to build or prepare does not authorize model calls. Raw payloads and responses remain ignored under `artifacts/`.

## Result

The authorized bundle completed with four ModelRun/fp4 calls, zero environment actions, and `$0.01351812` reported cost. All request and response hashes verified.

| Condition | First selection | Semantic result |
|---|---|---|
| C0 | DOWN | Preserved player-to-target navigation |
| C1 | Python | Called the extent change drastic; considered shape/size change |
| C2 | DOWN | Used centroid as movement confirmation; ignored shape change |
| C3 | DOWN | Ignored the explicit non-rigid relation; preserved target navigation |

Verdict: `replicate_c1_extent_before_mechanism`.

C1 is candidate-potent because it alone caused non-rigid consideration. It did not explicitly reopen LEFT semantics or choose an action-semantic discriminator. One stochastic sample is insufficient for a causal claim. The licensed next move is two additional matched C0 responses and two C1 responses at this same branch, counterbalanced in order, before engineering any component-extent producer.

An independent anonymous review also selected only C1 as candidate-potent and agreed that it considered a non-rigid transformation without reopening action semantics or choosing an action-semantic discriminator. The sole scoring disagreement was whether C1 preserved target navigation (`unclear` primary, `no` blind); this does not affect the replication decision.

Reviewed evidence is in `evidence.json` and `semantic-review.json`.
