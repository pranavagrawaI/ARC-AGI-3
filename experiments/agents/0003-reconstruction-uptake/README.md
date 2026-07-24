# Reconstruction Uptake Replay

> **Status: INVALIDATED PROTOTYPE — frozen 2026-07-21.**
>
> This implementation is preserved as a negative control: its executable API
> compliance criteria can be satisfied without prospective reconstruction.
> Do not repair or use it to license reconstruction claims. The corrected
> evidence/decision architecture is experiment `0004-counterexample-gated-ledger`.

Offline, fail-closed replay harness for testing whether an executable reconstruction treatment was actually instantiated.

## Dry run

```bash
uv run python3 experiments/agents/0003-reconstruction-uptake/run_replay.py \
  --trace /path/to/trace.jsonl \
  --output /tmp/reconstruction-uptake-dry-run \
  --first-step 1 \
  --last-step 17 \
  --condition required \
  --dry-run
```

A dry run validates the contiguous trace window and writes a replay plan containing current-observation hashes and recorded actions, but no next observations. It makes zero model calls and zero environment actions.

## Model execution

Model execution is intentionally double-gated:

```bash
uv run python3 experiments/agents/0003-reconstruction-uptake/run_replay.py \
  --trace /path/to/trace.jsonl \
  --output /tmp/reconstruction-uptake-required \
  --first-step 1 \
  --last-step 17 \
  --condition required \
  --execute-model \
  --confirm-model-cost I_ACCEPT_OFFLINE_MODEL_COST
```

It writes:

- `manifest.json`
- `events.jsonl`
- `compliance.json`
- `request-evidence.jsonl`
- `workspace/`

This path calls the model endpoint but never constructs an ARC environment and never sends a game action. Do not use it for confirmatory evidence until `experiment/protocol.md`, prompts, trace digest, and analysis contract are frozen.

## Tests

```bash
uv run python3 -m unittest experiments.agents.tests.test_reconstruction_uptake
```
