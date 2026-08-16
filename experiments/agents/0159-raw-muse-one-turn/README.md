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
settled grid ----------- CodeAct only ----> one palette-mode PIL `image`
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
action, and record the transcript. CodeAct variants receive one palette-mode
PIL `image` built from the settled public grid. The same object supports
`show(image)`, ordinary PIL operations, and exact `np.asarray(image)[y, x]`
cell inspection. Muse receives the
accumulated NOOA conversation. Provider-returned reasoning is replayed on the exact
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
- `codeact`: the same conversation plus the exact current settled grid and a
  host-generated copy of every executed settled-grid transition. Bounded NOOA
  CodeAct can use NumPy in a fresh, network-disabled, filesystem-confined
  process before returning one decision.
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
context is limited to `image`, the other current inputs, `np`, `show`, and
`return_result`;
worker helpers, generic sub-LLM fan-out, and `self` are absent. `codeact_notes`
adds only four brokered `self.*_notes` methods; direct filesystem access and all
other `self` access remain blocked. Note names are confined to
`[a-z0-9_-]+.md`, with 24,000 characters per file and 72,000 total. Generated
cells cannot execute game actions or mutate host evidence. Attempts to access
files, dangerous reflection, or dynamic execution primitives are blocked and
logged. NOOA's own
input prefill is identified separately and may perform harmless type inspection,
so the exact current image and trajectory are presented before the first
decision call. `image.size` and `image.crop(...)` use PIL coordinates;
`np.asarray(image)[y, x]` returns the canonical integer cell value.

There is still no claim protocol, verifier, supplied action map, world model,
planner, action batching, host-selected action, or semantic correction. The
compact variant adds retrieval and a model-authored checkpoint, but validation
checks structure and provenance rather than whether its game interpretation is
true. The same NOOA `Agent` object lives for the entire game. IPC is necessary
because the ARC SDK uses Python 3.14 while pinned NOOA uses Python 3.12.

## E0 typed host evidence

The cumulative 0160 worktree adds E0 beneath every existing variant without
changing Muse's visible tools or decision context. The Python 3.14 host now
persists `evidence/ledger.jsonl`, content-addressed public-grid blobs, immutable
evidence snapshots, and one atomic current-snapshot pointer.

E0 classifies only public host facts. It keeps boundary, outcome, and animation
as independent channels; uses separate `attempt_before` and `attempt_after`
identifiers for evaluator-local replay segmentation; and never infers a reset,
terminal outcome, object, or mechanic from pixels. The authoritative input
prefix and derived transition manifest have independently reproducible SHA-256
commitments. Attempt identifiers and the complete E0 projection remain hidden
from Muse in E0.

The implementation and contract are documented in
`e0_evidence.py`, `EXECUTABLE-THEORY-LANE-SPEC.md`, and `PROVENANCE.md`.

## WM0 commitments and WM1 quick theories

The cumulative `codeact_notes_compact_world_model` variant preserves the same
persistent investigator, notes, archived-history interface, human action labels,
and one-action host boundary. It adds three brokered affordances:

- `self.read_world_model()`;
- `self.write_world_model(source)`;
- `self.edit_world_model_function(name, source)`.

The rejected WM2 and WM2-C prompt treatments remain documented in
[`WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md`](WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md)
and [`WM2-C-GUIDED-FIRST-USE-SPEC.md`](WM2-C-GUIDED-FIRST-USE-SPEC.md). Their
runtime modules and flags were removed after live contact showed that escalating
from examples to opportunity cards and guided first use consumed attention
without producing an executable commitment. The executable-theory interface is
now passive: its fixed prompt explains why rendered Moore machines can help,
and Muse alone decides whether and when to read or edit the lane.

WM1 makes the first executable commitment smaller without removing WM0's full
Moore machine. Muse may define only `predict(grid, action)`, return `None` for
an unsupported action, and use `-1` for unclaimed cells. The evaluator scores
only claimed cells on ordinary transitions. When hidden state becomes useful,
Muse can replace that source with the full `State` / `init_state` / `transition`
/ `render` / `outcome` contract; no host component performs that promotion.

The initial quick source is a template, not an active belief. Every Muse edit is durably stored
as a candidate. Syntax, authority, contract, and containment failures leave the
prior current version unchanged. Any operationally valid candidate becomes the
current Muse-authored commitment even when replay contradicts it; the evaluator
reports separate initial-render, transition, and outcome measurements but never
rewrites source or recommends a repair.

Candidate execution occurs in a fresh Python 3.12 process with a sanitized
environment, closed inherited descriptors, Landlock filesystem confinement,
seccomp network denial, and CPU, memory, process, descriptor, output, wall-time,
and file-size bounds. The process receives only an immutable copy of the
candidate and the exact E0 evidence/blobs required for replay. Fourteen frozen
containment probes cover credentials, process/IPC access, path and mutation
escape, network/process creation, resource exhaustion, reflection, and a
transition-only hang.

WM0 passing its development gate does not establish behavioral improvement or
causal use. `P0` prospective shadow prediction is still absent. Bounded live
first contacts with Muse and GPT-5.6 Luna both preserved the actor loop but
produced no executable candidate; their frozen evidence is under
`artifacts/wm0-first-contact-v2/` and `artifacts/wm0-luna-first-contact-v1/`.
The Luna run uses `--model-profile gpt-5.6-luna`, which pins the OpenRouter
model and OpenAI-only provider route without changing the model-visible WM0
surface.

Both first contacts independently attempted ordinary pixel inspection through
the display-oriented NOOA transport wrapper and failed on `image.size`; Luna
also found that `np.asarray(image)` is zero-dimensional. The subsequent
interface repair keeps that wrapper private and exposes one 64x64 palette-mode
PIL `image`, without adding segmentation, object labels, or game semantics.

The WM1 quick-theory first contact is preserved under
`artifacts/wm1-quick-theory-muse-v1/`. It was valid for eight actions but created
no candidate and completed no level. Muse attempted `self.read_world_model()`
at turns 0 and 1; both bare-expression cells returned empty stdout because the
CodeAct cell behaves as a script rather than a REPL. The next interface boundary
was therefore automatic visibility of broker results, not another unchanged
paid run. The worker now auto-displays a final bare broker call and returns a
structured feedback envelope naming status, phase, code, state effect,
diagnostics, and the explicit retry surface. See the run-local
`AFTER-ACTION.md`.

The subsequent recovery-feedback check is preserved under
`artifacts/wm1-recovery-feedback-muse-v1/`. Automatic display was exercised by
a bare notes call and returned visible `[]`, closing the specific script-versus-
REPL defect. Muse nevertheless made zero world-model reads or edits and created
no candidate across eight valid actions. This leaves discoverability and low
immediate utility as competing explanations; the same surface should not be run
again unchanged.

The matched-surface Luna check under
`artifacts/wm1-recovery-feedback-luna-v1/` also produced zero world-model calls
or candidates. Luna used only ten analysis cells and chose a more varied route
from transition evidence, so its non-use may reflect low artifact utility rather
than inability. Cross-model non-adoption weakens a Muse-only explanation but
still does not separate discoverability from utility.

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
