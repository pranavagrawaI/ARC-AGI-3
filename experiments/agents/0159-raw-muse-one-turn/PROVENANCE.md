# Executable Theory Lane provenance

## Baseline

The cumulative experiment begins at ARC repository commit
`f3d8b3027508167a41a3de4ba5b125fec130575e`. The frozen 0159 worktree remains
unchanged; implementation occurs on `experiment/0160-executable-theory-lane`.

The existing persistent worker continues to pin NOOA at
`55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`. E0 imports no NOOA source and does
not alter the worker capability surface.

## Decision-local PIL observation

CodeAct receives one palette-mode PIL image rather than NOOA's transport-only
media wrapper plus a second NumPy grid. The worker's copied 16-color palette is
from `arc_agi.rendering.COLOR_MAP` in `arc-agi==0.9.8`; the installed
`rendering.py` SHA-256 is
`ef62e519ecbca857dfeb9956c8a6468335e7078b0e83e86324e5272caedacdba`.
`test_worker_palette_matches_authoritative_arc_rendering` parses the worker
constant without importing the Python 3.12 runtime and requires exact parity
with the authoritative Python 3.14 host package.

## E0 evidence adapter

`e0_evidence.py` is a new 0159-specific adapter. Its event schema is original
local code because the authoritative ARC host fields and the accepted
`attempt_before`/`attempt_after` contract do not match an upstream Tycho file
verbatim.

Its durable canonical-JSON, atomic-write, JSONL-append, `fsync`, and
content-digest mechanics are adapted from the repository-local implementation:

- source: `experiments/agents/0158-nooa-tycho-capability-build/host/canonical.py`;
- source repository commit: `f3d8b3027508167a41a3de4ba5b125fec130575e`;
- source SHA-256:
  `dd16d1574609125bff0a710471b4b6989e04f491019a12d9fc2fbed57f5a8379`;
- adaptation: helpers are private to the E0 module, immutable snapshot/blob
  writes are added, and the committed schema is `0159-host-evidence-v1`.

Tycho's separation of ordinary transitions, terminal evidence, reset/death
boundaries, and animation side channels informed the taxonomy:

- Tycho pin: `f68912a764372ead0a610db2e1c011d41ce5197e`;
- inspected source: `tycho/workspace/workspace.py`;
- pinned source SHA-256:
  `90a6a9d9060613c211708c2a99148acdb6754658e776d65556be670c43d796d3`.

No Tycho function is copied verbatim into E0, so source-parity is not claimed.
Conformance is instead frozen by
`tests/fixtures/e0-event-taxonomy-v1.json`, while independent commitment,
append-only frontier, animation-channel, reset-boundary, and host-integration
tests cover the adapted contract.

## WM0 executable-theory lane

WM0 is an adaptation rather than a directory copy of Tycho. The retained Tycho
mechanisms are the editable Moore-machine contract, seed-template posture,
candidate/version separation, replay against immutable evidence, UNKNOWN-cell
coverage, bounded observation variants, and distinct dynamics/outcome channels:

- Tycho pin: `f68912a764372ead0a610db2e1c011d41ce5197e`;
- `tycho/workspace/version_store.py` SHA-256:
  `448c0806f7c1e30767a64a2aa05ed61b700e8c6bf0c687068c66dc9177831180`;
- `tycho/workspace/wmlib_template.py` SHA-256:
  `d73db57154a8fc113fcb492d5966f7e96f2b2991489a139a72691394ac7cf8bd`;
- `tycho/workspace/templates/seed_world_model.py.tmpl` SHA-256:
  `d383f8efae32e0428e6401a97615e10f43f13c70d13d422d95a7fd6a5403236c`;
- `tycho/workspace/sandbox.py` SHA-256:
  `766c8e2fca38751c561c08ea8136d6e79d75df87b1e2d1ce0333f92643f43e89`.

The adaptation intentionally removes Tycho's planner, action generator,
subgoals, heuristic, builder, generic shell/file workspace, and procedural
repair guidance. It changes Tycho's SDK-style action representation to 0159's
validated human labels and changes Tycho's workspace-shaped evidence reader to
the E0 snapshot contract. No Tycho function is copied verbatim, so whole-file
or line-level parity is not claimed. Contract tests instead freeze activation,
replay, UNKNOWN coverage, terminal/reset segmentation, divergence blobs, and
function-versus-whole-file transaction equivalence.

The isolated evaluator imports the pinned NOOA Linux enforcement machinery
directly rather than approximating it:

- NOOA pin: `55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`;
- installed `nooa/runtime/sandbox/guards.py` SHA-256:
  `41bf5d21d832bba055551d65feef5359f3fd7fe71806084a0903e69f2b9953b0`;
- installed `nooa/runtime/sandbox/config.py` SHA-256:
  `221e4cca97b14893983c1bdbfa3247a9567217d64ea84b4dab3ac731e16a39a7`.

WM0 constructs a narrower `ResolvedSpec` than ordinary CodeAct: it does not
grant `/proc`, the persistent worker, provider state, IPC descriptors, or the
canonical lane. The canonical broker and evaluator protocol are new local code
in `world_model_lane.py` and `world_model_evaluator.py`. The 14 acceptance probes
live in `nooa-runtime/tests/test_world_model_lane.py`.

## WM1 quick commitments and recovery feedback

WM1 is new local code layered on the same broker and isolated evaluator. It
adds the `predict(grid, action)` partial-prediction contract, retains the full
Tycho-derived Moore-machine contract as an explicit promotion path, and emits
structured phase/code/state-effect/retry feedback. `-1` denotes an unclaimed
cell and `None` denotes whole-action abstention. The evaluator's per-transition
claim records are mechanical replay facts; they do not select or rewrite a
belief. Tests exercise partial support, contradiction with activation,
abstention, recovery after operational rejection, and later Moore promotion.
The fixed model-facing rationale for rendered Moore machines is a concise local
adaptation of the Tycho authors' explanation: decision-frame models may require
hidden state, state transitions, rendered observations and outcomes, and
transition emissions for informative animation frames.

## WM2 taught practice and workbench

WM2 is original 0159-specific interface code governed by
`WM2-GUIDANCE-AND-OPPORTUNITY-SPEC.md`:

- `nooa-runtime/theory_guidance.py` freezes a game-general, bounded Theory
  Practice Guide and complete synthetic examples. Executable examples pass the
  real WM1 broker and isolated evaluator.
- `nooa-runtime/theory_workbench.py` deterministically projects immutable E0
  evidence and a narrow frozen lane snapshot into exact transition facts and
  plain-language interpretation cards. Its closed condition registry exposes
  no game semantics, attempt identifiers, action recommendations, or automatic
  edits.

The guide and workbench have independent host/worker flags and source hashes.
When either is enabled, the governing WM2 specification is also included in the
run's source commitment. Audit records preserve rendered-content digests,
condition/card codes, cited evidence IDs, omissions, and byte counts without
claiming that exposure caused a model edit or action. P0 shadow prediction is
not part of WM2 and remains unimplemented.

## WM2-C guided first use

The first WM2 live run is preserved at
`artifacts/wm2-taught-theory-workbench-muse-v1/`. Its eight workbench views
surfaced repeated-action evidence, but Muse issued no theory-lane read or edit
and instead used 59 ad hoc Python cells. `nooa-runtime/guided_first_use.py` is a
new, bounded delivery module for the follow-up treatment. It maps only the
frozen `repeated_action_evidence`, no-active-model, and never-read state to an
exact read-only call and explicit edit/notes/defer choices. It contains no game
semantics, performs no tool call, and does not gate the environment action.
