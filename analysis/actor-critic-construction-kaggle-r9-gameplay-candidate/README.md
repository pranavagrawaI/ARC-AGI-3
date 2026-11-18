# R9 gameplay — official Agent.main, paired arms, one pinned public game per identity — locally verified, not frozen, not launched

## Identities and packages (one candidate, two builds)

| game | identity dir | package | play | freeze use under frozen R6 |
|---|---|---|---|---|
| s5i5-18d95033 | `analysis/actor-critic-construction-kaggle-r9-s5i5-20260915T183130Z/` | `deployment/kaggle/actor-critic-construction-kaggle-r9-s5i5/` | CLICK only (`available_actions [6]`), 8 levels | freeze produced, **never queried** — `CLICK x y` never equals `CLICK` in `available_actions` (frozen `live_agent.py`) |
| cd82-fb555c5d | `analysis/actor-critic-construction-kaggle-r9-cd82-20260915T183130Z/` | `deployment/kaggle/actor-critic-construction-kaggle-r9-cd82/` | directions + ACTION5 + CLICK, 6 levels | full lifecycle (query → assimilate → bound) |

`build.py` builds both from this directory: `gameplay_play.py` gets `PLAY_GAME = '<game>'  # BUILD-PARAMETER`,
`EXPERIMENT-CARD-<game>.md` becomes the package card, `MANIFEST-<game>.json` mirrors the package manifest.

## What changed since frozen R8 (both packages)

- `game_registry.py` (new): the R4 gate generalized over a pinned table {ls20/9607627b, s5i5/18d95033,
  cd82/fb555c5d}, every entry bound to public manifest 842477b6… by size and sha256; a game absent from the
  table cannot be selected. `game_preflight.py` (R4) is byte-identical and still runs in bootstrap.
- `gameplay_play.py`: `GAMES = (ls20, s5i5, cd82)`; the played game comes from the registry; each arm's
  solver session directories are recorded (`sessions`: dir name, ipc file names, frozen, host-events and
  constructor RESULT paths) so every per-arm file is addressable by exact path; `requests.jsonl` /
  `logical-requests.jsonl` carry receipts (identity, bounds, tools, sha256 + size of messages) with the
  full bodies per call under `raw-requests/`, keeping every jsonl under the 16 MB retrieval bound.
- Everything else byte-identical to R8: R5 contract, R6 freeze use, R7 reap, R8 supervisor, accounting
  (purse 258 + 1 warmup = 259; arm purses 129/129), envelope (600 s / 200 actions per arm, 1800 s launcher).

## Exact public bytes (item 1 of the R9 dispatch)

Fetched with `kaggle competitions download arc-prize-2026-arc-agi-3 -f environment_files/<game>/<version>/<file>`
into `analysis/actor-critic-construction-kaggle-verification-r9/public-assets/files/`, receipts beside them,
recomputed locally (`VERIFICATION.json`): s5i5.py 168,955 B `18d950333a006e98…`; s5i5 metadata 302 B
`1b4f3bfa7472ae0d…`; cd82.py 34,273 B `fb555c5ddf0a3b74…`; cd82 metadata 292 B `ccfe95990d5a9c5d…`. All
four match the manifest. (First attempt 404'd four times: a zsh word-splitting bug in my loop appended a
stray token to `-f`; the literal-path retry succeeded. The control download of ls20 metadata worked throughout.)

## Verification evidence

- `FINAL-TEST-s5i5.{stdout,xml}`, `FINAL-TEST-cd82.{stdout,xml}`: 105 passed each under mapped-root
  unshare with `PYTHONDONTWRITEBYTECODE` unset (R8's 102: 100 preserved, r8-specific delta/loader cases
  replaced by r9 ones; +registry equivalence, +session addressability). The suite selects the package with
  `R9_PACKAGE_GAME={s5i5|cd82}` and runs once per package.
- `test_runtime.py::test_actual_supervisor_worker_http` ×6 per package: the real official loop over both
  arms against the staged exact bytes of the package's game with a game-aware scripted model server
  (CLICK for s5i5); arm-equality; recording/scorecard reconciliation; construction only in B; session
  directories addressable; receipts digest-bound to the stored bodies and under the size bound.
- `test_gameplay_arms.py`: frozen parameters, arm-diff refusal, budget, purse, baseline-handed-the-constructor
  control, registry ↔ R4 gate equivalence on ls20 (and tamper refusal), session addressability.
- `VERIFY.stdout` / `LOCAL-CHECK.json`: both packages passed — 60 assets, 53 compiled, 422 protected,
  1089 predecessor bindings (R8 trees appended, never rewritten), 14 public-identity sources.
- `ARM-DIFF-s5i5.json`, `ARM-DIFF-cd82.json`; `LINT.stdout` clean; `TEST-INVENTORY.json`.

```sh
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-r9-gameplay-candidate/build.py
R9_PACKAGE_GAME=s5i5 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-r9-gameplay-candidate/tests
R9_PACKAGE_GAME=cd82 unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/actor-critic-construction-kaggle-r9-gameplay-candidate/tests
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/actor-critic-construction-kaggle-r9-gameplay-candidate/verify.py
```

## Not done / limits

No freeze, manifest, review, or push for either identity; each gate HOLDs on its missing manifest. Real
SGLang/GPU, 600 s tempo on these games, GAME_OVER handling on s5i5 under live play, and the incumbent critic
under live play are unexercised locally. The S5I5 pairing is a weaker contrast (overlay with an unused
freeze vs baseline); extending host-owned freeze use to `CLICK x y` is an R6 policy change, not done here.
