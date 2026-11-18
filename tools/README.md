# Tools

## Experiment evidence checks

[`experiment_checks.py`](experiment_checks.py) checks Flash-Next invalidators,
batch-permission evidence, fixture context coverage and non-overwriting artifact
preservation. See [commands, rule coverage and limits](EXPERIMENT-CHECKS.md).
Missing evidence stays unresolved; a completed probe cannot override an OOM.

## Failure-atlas packets and offline branch replay

`failure_atlas.py` converts a preserved Tycho-NOOA run into physically separate
pre-decision (`admissible/`) and successor/evaluator (`oracle/`) evidence. It
then restores each saved world model and replays the recorded prefix plus each
instantiated legal next action against a fresh local ARC engine. Neither command
calls a language model or consumes an ARC attempt.

Run it with the Python 3.14 ARC-host environment:

```bash
arc-host/.venv/bin/python tools/failure_atlas.py mine \
  runs/<run-id> analysis/failure-atlas/<run-id>

arc-host/.venv/bin/python tools/failure_atlas.py replay \
  analysis/failure-atlas/<run-id> analysis/failure-atlas/<run-id>-replay \
  --environment-files environment_files
```

Use `--sequence N` (repeatable) or `--max-packets N` for a bounded screen.
Replay requires full-prefix parity before scoring. The one documented
normalization is the initial `full_reset` event marker, which differs between
the ARC SDK's online and offline reset wrappers and is retained in the parity
report; grids and all other public metadata remain exact. Reports keep canonical
render accuracy, claimed-cell errors, unknown coverage/penalty, changed-cell
precision/recall, variants, and termination separate. CLICK is only branched at
coordinates already grounded by the recorded action or saved model; the report
marks the rest uninstantiated rather than pretending exhaustive coverage.

The maximum claim is deliberately narrow: saved-model one-step mechanics at
parity-checked visited states in one committed local engine build. It does not
establish actor uptake, gameplay benefit, or transfer.

Runs created before decision checkpoints existed can still screen their final
saved model over historical states, but packets mark it
`retrospective_final_candidate_not_historical`. Do not attribute that model to
the earlier decision. New runs use exact pre-call checkpoint workspaces.

## Zero-call builder-refresh inventory

`failure_atlas.py builder-refreshes` mines the exact builder and following
actor checkpoints already on disk. It totals builder cost and tokens, compares
each structured report with the previous report in that run, records the
actor's immediate first tool call, and joins any existing decision-point shadow
result. It makes no model call and contacts no game:

```bash
arc-host/.venv/bin/python tools/failure_atlas.py builder-refreshes \
  runs/<run-a>/game/decision-checkpoints \
  runs/<run-b>/game/decision-checkpoints \
  --shadow-root analysis/shadows \
  --output analysis/builder-refreshes/<inventory>.json
```

Candidates are ranked with unshadowed refreshes first, then exact report
repeats, missing action recommendations, unchanged recommendations, and changed
recommendations. Ranking identifies where a later matched shadow could buy the
most information; it does not establish that a refresh was causally redundant,
that deleting it is safe, or that gameplay would improve.

## Zero-call epistemic funnel

`epistemic_funnel.py` joins the five preserved ar25/LS20 development traces into
one decision-boundary census. It distinguishes model-authored hypotheses from
tool and host evidence, reports missing spend as unknown, deduplicates the first
supported failure per trace, and exposes the provisional-goal obligation horizon
instead of silently fixing it.

```bash
arc-host/.venv/bin/python tools/epistemic_funnel.py \
  --goal-obligation-actions 3 \
  --output analysis/epistemic-funnel/20260821-development-corpus
```

The command first runs scripted positive controls, then emits canonical JSON,
one-row-per-decision CSV, an answer-first Markdown report, and a self-contained
interactive HTML view. It makes no language-model or ARC environment calls.
Legacy ar25 call costs are sequence-aligned by tool-call IDs; Gemma uses exact
decision checkpoints. Retrospective branch replay remains explicitly separate
from historical first-failure attribution.

## Architectural-risk observations

`prediction_check.py` measures three narrow trace facts motivated by the risk
register in `Questions and Ideas/`. It does **not** score the theory documents as
successful predictions: the source risks and the current mechanisms are not always
the same, and checkpoint presence is not run validity.

```bash
arc-host/.venv/bin/python tools/prediction_check.py --corpus \
  --manifest analysis/prediction-check/20260821-corpus/corpus-manifest.json \
  --output analysis/prediction-check/20260821-corpus/prediction-check.json
```

- `superseded_model_unreachability` enumerates checkpointed `world_model.py`
  bodies and reports whether content-identical copies remain on any **final live
  workspace path**. It explicitly does not decide whether a lost body was valuable.
- `declared_tool_activity` separates tools offered, attempted in model responses,
  completed, and errored. Reading each `response.json` prevents a final tool call
  from being right-censored.
- `goal_lifecycle_activity` reads actual goal-tool events plus the final typed-goal
  artifact. It reports no-goal, frozen-after-proposal, or advanced lifecycle state;
  raw identifier occurrence is not treated as model attention.

The reviewed manifest admits or excludes every discovered run and records why.
Unknown runs force `status=review_required` and exit 2 rather than silently joining
the tally. Named run directories can be inspected without a manifest. The command
makes no language-model and no ARC environment calls; it establishes neither actor
comprehension, gameplay benefit, nor transfer.

## Hermetic pre-spend gate

`hermetic_gate.py` rejects orchestration and evidence-integrity regressions
without calling a language model, importing the ARC engine, or consuming an ARC
attempt. Run the generated fault campaign before considering a paid run:

```bash
arc-host/.venv/bin/python tools/hermetic_gate.py campaign \
  --output analysis/hermetic-gate/campaign.json
```

The deterministic campaign starts from a valid two-action trace, injects
worker restarts, ordering/count errors, illegal actions, evidence regression,
broken and cross-lane tool joins, failed action commits, malformed builder
advice, and missing completion repair. Every corruption must produce its
declared violation. The report also checks that JSON round trips and bijective
tool-call ID renaming preserve the verdict, and greedily reduces each failure
to a small evidence packet.

Audit any preserved run with the same contracts:

```bash
arc-host/.venv/bin/python tools/hermetic_gate.py audit runs/<run-id> \
  --output analysis/hermetic-gate/<run-id>.json
```

The audit checks public turn/decision/result continuity, action legality and
traceability, monotone public progress, actor/builder tool ownership, builder
trigger and completion-checkpoint lifecycle, planner trace ordering, and the
content commitments and blobs in decision checkpoints. Exit status is zero
only for a passing gate. A pass permits the next evaluation tier; it does not
establish LLM understanding, gameplay benefit, or transfer.

## Blind Human Play

Use `play_game.py` to play any ARC-AGI-3 game through the public frame/action
interface and save first-contact human data.

```bash
.venv/bin/python tools/play_game.py --game ls20
```

For games not already downloaded locally, use normal mode with `ARC_API_KEY`
loaded from `.env`:

```bash
set -a
source .env
set +a
.venv/bin/python tools/play_game.py --game ar25 --mode normal
```

Each run writes a timestamped folder under `human-play/<game>/` containing:

- `trace.jsonl` - admissible frame/action trace readable by `experiments.harness.viewer`
- `notes.md` - blind-play notes template plus live `n <note>` entries
- `summary.json` - compact run metadata
- `scorecard.json` - ARC scorecard snapshot
- `raw-recordings/` - ARC toolkit recording with full public frames

Commands during play:

- `1`..`7` - public ARC actions
- `6 <x> <y>` or `click <x> <y>` - coordinate action when available
- `r` or `reset` - reset through the local wrapper
- `c` or `coords` - toggle display-pixel coordinate rulers
- `i`, `inspect`, `tiles`, or `palette` - list visible same-color components with click coordinates
- `n <text>` - append a timestamped note
- `s` - print status
- `h` - print help
- `q` - quit and save

Coordinate actions use the public 64x64 display frame. The compact terminal
renderer compresses two vertical pixels into one text row, so coordinate mode
labels rows as `00/01`, `02/03`, and so on.

For games with clickable color tiles, `inspect` is usually easier than counting
against the ruler. It prints rows like `color=4 ... -> click 47 6`; paste the
`click x y` command to send `ACTION6` at that visible component.
