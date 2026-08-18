# ARC host

`arc-host` is the stable Python 3.14 environment side of the solver boundary.
It knows the ARC SDK and nothing about NOOA, Tycho, model providers, prompts, or
experiment numbers.

```text
ARC SDK                  arc-host                    solver process
--------                 --------                    --------------
raw frame     -> settled public Observation -> prefixed JSONL `turn`
GameAction    <- validated human ActionDecision <- prefixed JSONL `decision`
next frame    -> immutable Transition ---------> prefixed JSONL `action_result`
```

## Contract

The host owns:

- ARC environment creation, reset, stepping, and lifecycle;
- the final settled grid from each public frame sequence;
- translation between SDK IDs and `RESET`, `UP`, `DOWN`, `LEFT`, `RIGHT`,
  `SPACEBAR`, `CLICK`, and `UNDO`;
- legal-action and click-coordinate validation;
- level and within-level turn counters;
- action, level, and terminal stopping boundaries;
- a complete host-visible JSONL transcript.

`RESET` is always projected into the public action vocabulary, matching the
solver contract even when the SDK omits action `0` from a frame's advertised
actions. A returned `full_reset` frame begins a fresh attempt at within-level
turn `0` with no previous decision.

The worker owns every game-specific judgment. One worker process survives for
the entire episode. It receives no ARC SDK object or integer action ID. `CLICK`
uses model-facing `(row, col)` and is translated only at the SDK boundary to
`{"x": col, "y": row}`.

The subprocess environment is allowlisted. Model credentials (`ARC_LLM_*`,
`OPENAI_API_KEY`, or `OPENROUTER_API_KEY`) may cross because they belong to the
worker; the environment credential `ARC_API_KEY` never does.

Every transport line is prefixed so incidental worker stdout cannot become a
protocol message:

- host to worker: `ARC_HOST\t{...}`;
- worker to host: `ARC_AGENT\t{...}`.

Protocol version 1 has this lifecycle:

```text
start_game -> ready
(turn -> decision -> action_result -> observed)*
finish -> finished
close
```

The `action_result` acknowledgement makes the executed successor observable to
the worker even when the host stops before another decision, such as at a
one-level boundary.

The host closes resources only when they expose a public `close()` method.
Cleanup errors cannot replace the primary run error. Both normal and exceptional
runs write `summary.json`; exceptional runs also append a structured `error`
record to `transcript.jsonl` with the progress observed before failure.

## Run

The command after `--` may use a different Python environment:

```bash
cd arc-host
uv run --frozen arc-host \
  --game ls20 --seed 0 --max-actions 44 --stop-after-levels 1 \
  --env-file ../.env \
  --run-dir /absolute/path/to/run \
  -- uv run --frozen --project /absolute/path/to/solver python -m solver.worker
```

This package is independently buildable and testable:

```bash
(cd arc-host && uv run --frozen ruff check src tests)
(cd arc-host && uv run --frozen pytest -s -q)
```
