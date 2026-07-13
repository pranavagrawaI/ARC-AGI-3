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

For hosted development, put `QWEN_API_KEY`, `QWEN_MODEL`, and
`OPENAI_BASE_URL` in the ignored repository `.env` (see `.env.example`). First
run the single-request compatibility probe:

Model Studio and Large Model Inference must first be activated in the same
region as the workspace endpoint and API key. Alibaba documents
`403 AccessDenied.Unpurchased` as an activation/eligibility failure; a
sub-workspace may also require explicit model-calling permission. See the
[activation guide](https://www.alibabacloud.com/help/en/model-studio/activate-alibaba-cloud-model-studio)
and [error-code guide](https://www.alibabacloud.com/help/en/model-studio/error-code).

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/agents/0002-llm-repl-agent/probe_model.py
```

It sends one tiny image and the Python tool schema, then prints only transport
metadata, token usage, and whether the model returned the expected tool call.
It never prints the key, response text, reasoning text, or tool arguments.

Then run a deliberately capped game smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
uv run python experiments/agents/0002-llm-repl-agent/run.py \
  --game ls20 \
  --env-seed 0 \
  --model-seed 0 \
  --max-actions 5 \
  --max-model-calls 15 \
  --max-reported-tokens 50000 \
  --no-retrospective
```

When `QWEN_MODEL` is present, the runner selects the explicit `qwen3.6`
profile: thinking and preserved thinking enabled, `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `presence_penalty=0`, and
`max_completion_tokens=4096`. Override the last value with
`--max-completion-tokens`; the profile rejects `--max-tokens` because that does
not cap thinking tokens.

The model-call limit is hard. The reported-token limit refuses the next request
after cumulative provider-reported usage reaches the threshold; a response may
cross that threshold because its input usage is not known locally in advance.
The per-request completion limit bounds output-side overshoot.

Each Python result sent back to the model is valid JSON capped at 4,000
characters. The result includes `truncated`, `original_chars`, and `limit`
metadata, and `process.jsonl` preserves the exact bounded text the model saw.
This is a generic context-bandwidth constraint; it does not summarize or
interpret the output for the model.

For the local FP8 checkpoint later, keep the same Qwen request profile and
change only the transport and served model ID:

```bash
uv run python experiments/agents/0002-llm-repl-agent/run.py \
  --model Qwen/Qwen3.6-27B-FP8 \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-profile qwen3.6 \
  --game ls20 --max-actions 5
```

Local endpoints never inherit hosted credentials from `QWEN_API_KEY` or
`OPENAI_API_KEY`; they receive `EMPTY` unless `LOCAL_MODEL_API_KEY` is set.

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
