# m2_guard: silent-zero guard for dfranzen-lineage notebooks

`build.py` applies four edits to a drawn parent notebook (F1x screen-2 v1 or the raw control) and leaves every other cell
byte-identical (`tests/test_guard.py::BuildTouchesOnlyAnchors`).

| edit | where | healthy run | fault |
|---|---|---|---|
| GPU check | server cell, after the torch check | prints the GPU name | any non-RTX-PRO-6000 device (T4, 2x T4, mixed, none) raises before the model load |
| supervisor | new cell before the pickle/solver import | polls `/health` until READY, then `proc.poll()` every 10 s; no other effect | if the server exits: killpg the old group, sleep 10 s, relaunch with the same args; at most 2 restarts, then `server_abandoned`; events in `guard-lifecycle.jsonl` |
| patience | same cell | `ARC3_MAX_ANALYZER_FAILURES` 10 → 40; the counter resets on any answered turn, so it binds only on consecutive failures | a game rides out ~17 min of failed turns (a ~10 min reload) instead of quitting after ~5 min |
| end guard | run cell, after `bm.run` | prints `M2 GUARD end {...}`; token totals include every action history record plus the final no-move turn | 0 levels and (0 generated tokens, or ≥ 90 % of served games ended on analyzer failure or crash) raises, so a dead stack becomes ERROR, not a valid-looking 0.00 |

Restarts are bounded at 2. There is no time-coded failure encoding: an ERROR's timing is whatever the fault produces.

Build: `python3 -B solvers/m2_guard/build.py [f1x] [control]` writes `kernel/<name>/m2-guard-<name>.ipynb`,
`experiment/fault-screen-1/kernel/`, and `manifest.json`. Repeated builds are permitted before launch.
Once `experiment/fault-screen-1/launch-attempt.json` exists, every build request refuses before any write, including
control-only builds; the frozen screen notebook, metadata and manifest remain unchanged.
Tests: `python3 -B -m unittest solvers/m2_guard/tests/test_guard.py`. They execute the guard text taken from the built notebook.
The end guard regression uses all three real zero-level F1x screen-2 games: 233,500 history tokens and zero final-turn
tokens. Separate synthetic fixtures verify final-turn addition and a dead endpoint whose only history is a zero-token RESET.

After the screen launch, the two build tests were repaired to exclude the real launch marker and downloaded outputs from
their temporary prelaunch fixtures. The original frozen test bytes are retained in
`evidence/postlaunch-test-repair-20261022/test_guard.launched.py.txt`; `repair.json` verifies the original ten launch inputs,
resolving only the historical test path through that archive. The original freeze, runtime notebooks, builder and card are
unchanged. The current test revision passes all 18 tests and is not the test revision named in the historical freeze.

CPU fault harness: `analysis/f1x-hidden-zero-diagnosis-20261021/faults/`. It runs the notebook's own env, config,
startup-wait and run cells in competition mode, against a local 110-game Arcade and a fake SGLang process. Prior completed
short runs and completed production-timing probes are retained under that directory's `runs/`; full raw outputs remain under
`/tmp/f1x-faults`. See `faults/FAULTS.md` for provenance limits and the separate real-record zero-level regression.
The local review fixes and red/green evidence are in `evidence/review-fixes-20261021/`.

Open risks (not provable on CPU):
- The real SGLang relaunch remains unqualified after `experiment/fault-screen-1`: loading resumed with restored free VRAM,
  but no second READY was observed before the screen ended. This does not establish port binding or complete recovery.
  `experiment/fault-screen-1/gate-evaluation-recovered-20261022.json` includes recovered GPU/end console evidence and keeps
  the overall NOT_QUALIFIED verdict.
- Recovery from a live but unusable server is unsupported: after first READY the supervisor checks process exit only.
  A health failure or rejected requests while the process remains alive does not trigger a restart. The end guard can reject
  an aggregate zero afterward when its conditions hold; the process-kill fault screen does not qualify recovery from this mode.
- A mid-run death after some levels keeps a partial score; this is deliberate. The end guard fires only on zero levels.
- The control parent is `analysis/m2-winners-20261001/runs/raw/dfranzen-m2-raw.ipynb` (sha in manifest.json). Before any
  control use, confirm it is byte-identical to the drawn dfranzen-m2-raw v1.
