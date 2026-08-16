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

There are four cumulative variants:

- `predict`: persistent conversation only. It has no tool loop and never receives
  the structured trajectory.
- `codeact`: the same conversation plus a host-generated copy of every executed
  settled-grid transition. Bounded NOOA CodeAct can use NumPy in a fresh,
  network-disabled, filesystem-confined process before returning one decision.
- `codeact_notes`: `codeact` plus the NOOA ARC solver's familiar
  `list_notes`/`read_notes`/`append_notes`/`write_notes` interface. Muse authors
  durable semantic knowledge under the run-local `knowledge/` directory; the
  host never supplies or revises game interpretations.
- `codeact_notes_compact`: an isolated continuation variant. It retains only a
  complete-turn trajectory suffix, exposes the credential-redacted immutable
  archive through the namespaced `history.*` object, and can apply forced or
  automatic prepare-validate-commit compaction at environment-turn boundaries.

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
planner, action batching, host-selected action, or semantic correction. The
compact variant adds retrieval and a model-authored checkpoint, but validation
checks structure and provenance rather than whether its game interpretation is
true. The same NOOA `Agent` object lives for the entire game. IPC is necessary
because the ARC SDK uses Python 3.14 while pinned NOOA uses Python 3.12.

Reasoning replay and context compaction remain separate boundaries in the three
original variants. `codeact_notes_compact` installs the mechanism behind a
separate flag.

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

The bounded consolidation and frozen-boundary comparisons are complete; see
`COMPACTION-COMPARISON.md`.

## Live compaction transaction

At the configured environment turn, `codeact_notes_compact` performs:

1. **Prepare:** copy the completed transcript, trajectory, provider log, and
   notes into `compactions/0001/frozen-run/`; generate one bounded checkpoint in
   a separate stateless provider call.
2. **Validate:** enforce the checkpoint schema, 16,000-character cap, stable
   evidence IDs, artifact paths, and SHA-256 hashes. Failure stops the run before
   changing active context or issuing an ARC action.
3. **Commit:** install the redacted history snapshot, collapse only complete old
   NOOA turn ranges, and insert an explicit `SAME GAME, SAME AGENT` Summary.
4. **Resume:** retain complete recent turns and the current observation, make
   older evidence available through `history.*`, and continue normally.

The live Python interface is:

```python
history.turn(turn)
history.get(event_ids)
history.turns(start, end, streams=None)
history.find(query, filters=None)
history.note(name)
history.provider_episode(sequence, phase)
history.around(event_id, before=1, after=1)
```

The bounded live seam run compacts before turn 5, retains three raw turns, and
stops after the turn-5 action. A prior live probe established that six raw
CodeAct turns already exceed the 131,072-token endpoint limit; the archive
still preserves and searches every earlier turn.

```bash
ARC_API_KEY=... ARC_LLM_API_KEY=... PYTHONPATH=. uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py \
  --variant codeact_notes_compact \
  --compact-at-turn 5 \
  --compaction-recent-turns 3 \
  --action-cap 6
```

Automatic mode reads the provider-reported prompt count only after a complete
environment turn. It divides by the usable input window (131,072 less the
8,192-token output reserve), triggers at 80%, and keeps the largest complete
suffix projected to place the next call at or below 30%. It never splits an
environment turn. Repeated compaction replaces the earlier active Summary while
the immutable archive remains cumulative.

For this variant, NOOA's built-in context-error archival is disabled. An
oversized in-loop provider request preserves the active event store, emits an
`implicit_context_archival_blocked` audit event, and returns
`needs_compaction` without executing an ARC action. In automatic mode the host
then runs the same validated compaction transaction and retries the same
environment turn once. Only that host transaction may delete active history;
a second overflow fails visibly rather than falling back to lossy archival.

The v24 failure-path probe deliberately raised the boundary trigger to
99.9999%. On turn 6 the provider rejected a prompt containing at least 122,881
input tokens plus the 8,192-token output reservation. NOOA attempted four API
retries, but every archival attempt was blocked without collapsing events. The
worker then returned `needs_compaction`; the host collapsed completed turns
0--4, retained turn 5 and the partial turn-6 work, and retried sequence 6. It
returned one DOWN decision and the host issued exactly one turn-6 action. The
retry used 63,070 / 122,880 tokens (51.3%): safe continuation succeeded, though
the retained partial failed-turn work means emergency recovery does not meet the
ordinary 30% post-boundary target.

The v23 live probe triggered after turn 5 at 104,185 / 122,880 tokens (84.8%),
collapsed turns 0--4, retained raw turn 5, and completed turn 6 at 32,336 tokens
(26.3%). Turn 7 then grew normally to 34.8%; it did not meet the 80% trigger.

```bash
ARC_API_KEY=... ARC_LLM_API_KEY=... PYTHONPATH=. uv run --frozen python \
  experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py \
  --variant codeact_notes_compact \
  --auto-compact \
  --action-cap 8
```

## Fixed live configuration

- game: online LS20, fresh seed-0 reset;
- model: `meta/muse-glimmer-30b`;
- provider for the compaction variant: qualified sticky ordered pool through
  OpenRouter: DeepInfra BF16, Together, Fireworks, then Phala;
- provider attempts: 90-second hard boundary, no provider-local retries; a
  failed provider is removed for the rest of the run and the successful
  successor becomes preferred;
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
- each pinned-provider attempt, timeout/failure, successful provider, and
  sticky failover;
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
