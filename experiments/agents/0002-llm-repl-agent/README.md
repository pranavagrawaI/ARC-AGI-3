# Generic LLM + Python Workspace Baseline

This experiment tests an LLM that invents structure during interaction. It is
deliberately not a hybrid and imports no option-agent ontology, factor state,
predicate, option, or planner.

The policy receives only:

- the rendered public frame;
- public state metadata and currently legal action IDs;
- its own public transition history;
- one fresh, persistent Python workspace for the current run.

The game identifier and benchmark name are absent from the model conversation
and workspace. Independent runs create fresh conversations and workspaces. An
optional run-local color permutation tests dependence on remembered color
conventions without changing the environment's mechanics.

## Run

Start an OpenAI-compatible multimodal server separately, then run:

```bash
OPENAI_MODEL=<checkpoint> \
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
uv run python experiments/agents/0002-llm-repl-agent/run.py \
  --game ls20 \
  --env-seed 0 \
  --model-seed 0 \
  --max-actions 500
```

Add `--color-permutation-seed 17` for a deterministic display-color renaming.
The actual game ID remains in the host-side `manifest.json` for experimental
bookkeeping but is never mounted into the Python workspace or sent to the model.

## Isolation and admissibility

Python executes in a `bubblewrap` sandbox containing only the host's isolated
Python 3.12 standard library, an empty run-local writable workspace, `/tmp`, and virtual
`/proc`/`/dev`. It has no repository mount, home directory, game implementation,
or prior-run files. Socket creation and connection are blocked by an inherited
seccomp filter. This managed WSL runtime cannot create a separate network namespace,
so the filter is tested explicitly rather than claiming namespace isolation. Every environment action still crosses
the public harness, including legal-action and coordinate validation.

The workspace uses `python -I -S` and mounts only `/usr/bin/python3.12` plus
`/usr/lib/python3.12`, rather than the project virtualenv. Mounting the project
environment would expose `arc_agi` and downloaded game implementation code to
generated Python. The manifest records the exact interpreter version, import
path, and discoverable top-level modules because the standard library still
belongs to the execution host rather than a pinned container image.

## Evidence

Each run writes:

- `trace.jsonl`: public action/observation trace;
- `process.jsonl`: model responses, Python calls, action/tool-indexed workspace
  changes, repairs, evictions, decisions, and transitions;
- `workspace/`: the policy workspace snapshotted before retrospective measurement;
- `manifest.json`: exact endpoint/model/seeds/budgets and aggregate token usage;
- `scorecard.json`: post-run score evidence unavailable to the policy;
- `process-annotation.json`: a human annotation template for discovery times,
  representation latency/stability, and wasted-action categories.

The diagnostic rubric is applied after the run. It is not included in the policy
prompt, because telling the policy which discoveries to report would itself be
representation scaffolding.

After reviewing `process.jsonl`, the policy workspace, and the action-indexed
workspace changes, fill every `unreviewed` entry in `process-annotation.json`.
Then validate and derive the compact measurements with:

```bash
uv run python experiments/agents/0002-llm-repl-agent/summarize_process.py \
  experiments/agents/0002-llm-repl-agent/artifacts/runs/<run-id>
```

This produces `process-measurements.json`. Subjective discovery judgments remain
human annotations; the script prevents blank templates from masquerading as
completed measurements.

Spatial or layout permutations are not included yet. A generic visual transform
is not behavior-preserving unless directional actions and coordinate actions are
transformed with a verified game-independent mapping; applying image-only layout
changes would measure interface corruption rather than adaptation.
