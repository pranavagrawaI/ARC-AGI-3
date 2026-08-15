# 0159 — Persistent Muse NOOA loop

0159 starts from the simplest continuous-conversation solver, then adds matched
capabilities one at a time: read-only programmatic analysis of recorded experience,
followed by model-authored durable beliefs. The directory name
is historical: the raw one-turn prototype was removed after this replacement
operated in a live LS20 run.

## Architecture

```text
Python 3.14 ARC host                         Python 3.12 NOOA worker

public frame -> PNG ---- JSONL IPC -------> one persistent NOOA Agent
recorded transitions -- treatment only --> Predict OR CodeAct analysis
                                               │
                              early decision ──┤
                                               │ 11-call analysis cap
                                               ▼
                                  action-only reserved commitment
                                               │
                                               ▼
                                      Muse Glimmer 30B decision
                                               │
legal-action check <------ JSONL IPC ----------+
        |
        v
exactly one ARC action -> returned frame sequence -> settled grid -> repeat
```

The host does only work that does not require model judgment: capture the public
frame, list legal human-facing controls, reject an illegal result, execute one
action, and record the transcript. Muse receives each image and the accumulated
NOOA conversation. Provider-returned reasoning is replayed on the exact
assistant message that produced it, so it remains available after tool results
and on later game turns while that event remains in active context. Muse returns:

```json
{"action": "DOWN", "x": null, "y": null, "explanation": "..."}
```

The model never sees ARC SDK integers. The host presents the same public control
vocabulary as the ARC Prize human interface: `UP`, `DOWN`, `LEFT`, `RIGHT`,
`SPACEBAR`, `CLICK x y`, `UNDO`, and `RESET`. A click is represented structurally
as `{"action":"CLICK","x":X,"y":Y,...}` with coordinates in `[0,63]`.
The host alone translates these labels to SDK identifiers at the environment
boundary. Only controls reported available in the current frame are legal.

There are three cumulative variants:

- `predict`: persistent conversation only. It has no tool loop and never receives
  the structured trajectory.
- `codeact`: the same conversation plus a host-generated copy of every executed
  settled-grid transition. Bounded NOOA CodeAct can use NumPy in a fresh,
  network-disabled, filesystem-confined process before returning one decision.
- `codeact_notes`: `codeact` plus the NOOA ARC solver's familiar
  `list_notes`/`read_notes`/`append_notes`/`write_notes` interface. Muse authors
  durable semantic knowledge under the run-local `knowledge/` directory; the
  host never supplies or revises game interpretations.

After every incomplete CodeAct round, the host appends the exact number of
analysis calls used and remaining. The messages become progressively more
salient at five, three, two, one, and zero calls remaining. The model may commit
early. If it consumes all eleven analysis calls, an action-only commitment
phase is reserved. It permits up to three validation attempts, each retaining
the complete active conversation and provider reasoning but exposing only
`return_result`, with no Python or analysis tools. The provider is asked for the
named tool, using Muse's native
`return_result(result={"action": ..., "x": ..., "y": ..., "explanation": ...})`
encoding. Because
DeepInfra has returned ordinary text despite named `tool_choice`, the adapter
also accepts one exact JSON Decision object. It parses the complete response as
JSON and applies the same strict Pydantic and legal-action checks; it never
extracts from prose, repairs fields, infers intent, or substitutes an action.
Malformed responses receive commitment-specific feedback naming both exact
encodings, not NOOA's generic JSON-output instructions. Three failed attempts,
or a malformed or illegal decision, is a model failure.

The `codeact` treatment has no writable workspace. Its model-visible execution
context is limited to the current inputs, `np`, `show`, and `return_result`;
worker helpers, generic sub-LLM fan-out, and `self` are absent. `codeact_notes`
adds only four brokered `self.*_notes` methods; direct filesystem access and all
other `self` access remain blocked. Note names are confined to
`[a-z0-9_-]+.md`, with 24,000 characters per file and 72,000 total. Generated
cells cannot execute game actions or mutate host evidence. Attempts to access
files, dangerous reflection, or dynamic execution primitives are blocked and
logged. NOOA's own
input prefill is identified separately and may perform harmless type inspection,
so the current image and trajectory are presented before the first decision call.

There is still no claim protocol, verifier, supplied action map, world model,
retrieval/indexing system, planner, action batching, host-selected action, or
semantic correction. Durable notes are model-authored beliefs, not verified
facts and not a host-generated summary. The same NOOA `Agent` object and event
history live for the entire game. IPC is necessary because the ARC SDK uses
Python 3.14 while pinned NOOA uses Python 3.12.

Reasoning replay and context compaction are separate boundaries. This revision
repairs exact replay; 0159 does not yet install a semantic summarizer. If the
model context limit is reached, NOOA's existing archival behavior may replace
older active events. A long run that crosses that boundary therefore cannot be
claimed to retain its complete reasoning history until compaction is specified
and tested separately.

## Offline continuation-checkpoint probe

`compaction.py` implements the first non-live slice of that separate test. It
does **not** change the worker's active context and it does not make a provider
call. Given a frozen run and a candidate checkpoint, it:

- treats `transcript.jsonl`, `trajectory.jsonl`, `provider-events.jsonl`, and
  `knowledge/*.md` as immutable evidence with stable event IDs;
- exposes deterministic `get`, turn-range, and lexical/filter retrieval over a
  redacted model-searchable projection;
- validates the candidate's schema, size, and evidence references without
  validating game semantics;
- writes a source manifest with paths and SHA-256 commitments, the checkpoint,
  episode index, advisory review, and an explicit recovery packet;
- retains a contiguous recent-turn suffix and clearly says that the context
  transition is neither a game reset nor an action result.

The archive also provides decision-shaped retrieval views discovered through
the frozen-boundary comparison:

- `history.turn_view(turn)` returns that turn's transcript plus mechanical
  per-value bounding-box motion, excluding provider noise and raw grids;
- `history.provider_episode(sequence, phase)` groups context recovery,
  commitment attempts, or transport failures causally;
- `history.note(name)` returns one exact durable note;
- `history.around(event_id, before, after)` returns a bounded same-stream
  neighborhood with explicit truncation.

On the v7 hard probe, checkpoint plus these structured views scored 15/15 in
three repeats versus 1/15 for suffix and checkpoint alone. Relative to raw/flat
retrieval, structured retrieval cut prompt tokens from 655,067 to 203,420 and
cost from $0.098192 to $0.037404. This is offline recovery evidence, not yet a
live gameplay or autonomous-retrieval result.

The human-authored
`examples/v14-reference-checkpoint.json` is a reference specimen for the frozen
turn-15 boundary, not evidence that a model can produce a good checkpoint. It
lets us test the representation and retrieval contract before comparing
compression policies.

```bash
uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/compaction.py \
  --run-dir experiments/agents/0159-raw-muse-one-turn/artifacts/20260815T-decision-transport-v14-codeact-notes-ls20-retry2 \
  --candidate experiments/agents/0159-raw-muse-one-turn/examples/v14-reference-checkpoint.json \
  --output-dir /tmp/0159-v14-compaction \
  --resume-turn 16 \
  --recent-turns 6
```

This deliberately stops before live integration. The next discriminating step
is to have a bounded consolidation call produce the same schema, compare its
recovery behavior with the current NOOA rebuild at the frozen boundary, and
only then install proactive compaction in the persistent worker.

## Fixed live configuration

- game: online LS20, fresh seed-0 reset;
- model: `meta/muse-glimmer-30b`;
- provider: DeepInfra BF16 through OpenRouter, no fallback;
- NOOA: `55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`;
- action cap: 44;
- CodeAct treatment: at most 14 provider calls per environment turn: up to 11
  analysis calls plus three action-only commitment validation attempts;
- exactly one environment action per turn in both variants;
- success: `levels_completed >= 1`.

## Run

```bash
ARC_API_KEY=... ARC_LLM_API_KEY=... uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py \
  --variant predict --action-cap 44
```

Run the matched trajectory-analysis treatment:

```bash
ARC_API_KEY=... ARC_LLM_API_KEY=... uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py \
  --variant codeact --action-cap 44
```

Run the next cumulative capability, durable model-authored beliefs:

```bash
ARC_API_KEY=... ARC_LLM_API_KEY=... uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py \
  --variant codeact_notes --action-cap 44
```

The Python 3.12 worker environment is locked separately:

```bash
uv run --frozen --project \
  experiments/agents/0159-raw-muse-one-turn/nooa-runtime \
  pytest -s -q
```

Each live run is non-resumable. Decision frames, returned animations, decisions,
action results, worker logs, and ARC recordings remain under the ignored
`artifacts/` directory. Every run
records SHA-256 commitments for the host, worker, worker lock, and reused shared
harness sources. Treatment and baseline artifacts must have the same source
bundle commitment to form a matched comparison.

The host uses the shared public-observation copier and terminal-state contract.
Worker protocol frames bypass mutable Python stdout state and are written through
a duplicated startup OS file descriptor, so sandbox or context-recovery output
capture cannot swallow a completed decision.
ARC's `frame` field is a temporal sequence: when an action returns multiple
grids, 0159 archives every numeric grid and PNG as a separate animation event,
then uses the final grid as the next playable decision frame. This follows the
SDK renderer and Tycho's decision-frame boundary; it neither merges nor silently
drops returned evidence. The model receives only the settled decision image,
not a host interpretation of the animation. 0159 records an action attempt
before crossing the remote step boundary; if acknowledgement is lost, the run stops as
`uncertain_environment_action` and is marked invalid instead of guessing whether
the action was consumed. Provider, model, worker, sandbox, and host failures
remain distinct. The summary commits the complete final trajectory.

### Provider diagnostics

`provider-events.jsonl` records UTC timestamps and monotonic durations for:

- turn receipt and NOOA generation start;
- provider-call start, message/image-size counts, and 30-second wait heartbeats;
- every scheduled retry, including status code, 429 classification, and backoff;
- provider completion, reported usage, or the terminal exception;
- CodeAct cell source, block decisions, and bounded output in the treatment;
- durable-note reads and writes by filename, character count, and content hash
  (with the exact model-authored content preserved under `knowledge/`);
- exact analysis-budget updates and reserved-commitment start/completion;
- exact model content, reasoning, and tool arguments returned during each
  reserved commitment attempt, including rejected attempts;
- model-visible tool names and the exact tool-choice constraint for each call;
- validated decision completion.

It does not record prompts, image bytes, or credentials. Outside reserved
commitment attempts, response logging remains counts and truncated error
descriptions. A stalled call with heartbeats but no retry event means the current
provider request has not returned. A retry event identifies the rejected or
failed attempt and its backoff explicitly.

## First live replacement run

`20260814T112819Z-nooa-ls20` proved the persistent loop over 30 executed actions
with one worker PID. It completed no level. Before action 31, the provider call
exceeded the original 900-second host wait, so the host terminated without
issuing another ARC action. The final host wait is 3,600 seconds so NOOA's
bounded rate-limit retries are not cut off by a shorter outer timeout.
