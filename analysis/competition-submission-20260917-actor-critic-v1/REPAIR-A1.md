# T3R — repair A1, refreeze and local requalification (2026-09-17)

Status: **locally requalified; not authorized.** No `final-review.json`, `authorization.json` or `CONSUMED.json` exists. Gate is HOLD / exit 2 solely because fresh independent review is absent (`checks/repair-a1/gate-hold.log`). No kernel push, competition submission, GPU, paid model call or provider write occurred.

Candidate: `pranavagrawai/arc3-ac-competition-20260917-v1` (identity, title, SDK slug, `kernel-metadata.json`, `budget.py`, caps 12/4 and the experiment card are byte-unchanged).

## What was repaired (reviewer fix A1 only)

| # | File | Change |
|---|---|---|
| 1 | `runtime/chassis/solver_agent.py` (`deep_investigate`, before `maximum = int(os.environ.get("ARC_DEEP_MAX_CALLS", "1"))`) | `if os.environ.get("ARC_GAME_REQUEST_LEDGER"): raise RuntimeError("deep investigation is disabled under competition request accounting")` — fail-closed: under competition accounting the tool raises before touching `self._llm`, so no ledger-less request, no `deep_investigations.jsonl`, no lifecycle gap. |
| 2 | `runtime/headless.py` Actor system prompt | Removed the sentence advertising `await self.deep_investigate(...)` for hard abstraction failures. No other policy text changed (diff is exactly the two source lines → `'off. '`). |
| 3 | `runtime/CHASSIS.json` | Rehashed by `runtime/freeze_candidate.py`: `contract_sha256` `03d0de3e…` → `9431c419…`; `files` hashes for the two sources above. |
| 4 | `submission.ipynb`, `ASSETS.json` | Rebuilt by `build.py`; only the embedded `nooa-join.zip` payload changed (`f42f8574…` → `3bccca35…`). Verified `json.loads(submission.ipynb) == build.notebook()` and payload hashes == `ASSETS.json`. |

Reviewer recommendation B was **not** applied: post-gameplay `validate_output` and the inherited end-of-run contracts still raise (unchanged semantics). `ARC_DEEP_MAX_CALLS`, `WORKER_ENVIRONMENT_KEYS`, `entry.py`, `play.py`, `budget.py` are untouched.

Permanent regression (package-owned, manifest-bound):

- `qualify_gateway.py --mode deep`: the scripted model's first Actor turn executes `r = await self.deep_investigate("Which rule moves the piece?", ["decision_packet"])` through the exact notebook bytes, official `Agent.main` and official REST gateway; the fixture then asserts the refusal text reached the model, no `deep_investigations.jsonl` exists, zero non-critic tool-less requests were served, and `server_requests_total == shared_ledger_rows == expected_launcher_solver_requests`, then runs the outer supervisor reconciliation and perturbation rejection as in every other mode.
- `test_local.py::test_deep_refusal_official_gateway`: runs that fixture (2 games) inside pytest on the pinned cp312 environment and re-asserts the summary plus the absence of the advertisement in `runtime/headless.py`.

## Before / after on the reviewer's own fixture (`independent-review/qualify_gateway_deep.py`, unchanged)

| | `checks/repair-a1/deep-before` (pre-repair notebook bytes, `910ac026…`) | `checks/repair-a1/deep-after` (repaired notebook bytes, `3f38ad9b…`) |
|---|---:|---:|
| Server requests | 14 | 13 |
| Shared ledger rows (Actor + critic) | 13 (11 + 2) | 13 (11 + 2) |
| `expected_launcher_solver_requests` | 13 | 13 |
| `deep_investigations.jsonl` records | 1 (`max_tokens 32768`, `enable_thinking: true`) | 0 |
| `deep_investigate(` seen in Actor tool calls | yes | yes (attempted, then refused) |
| Refusal text fed back to the model | no | yes (5 subsequent requests carry the tool traceback) |
| Both games `WIN`, scorecard closed | yes | yes |
| Supervisor `reconcile_launcher_request_count` | `RuntimeError … expected 13, observed 14` (`RECONCILE-ERROR.txt`) | passed, `whole_model_request_count_verified: true` |
| `RESULT.json.status` | `reconciliation_failed` | `passed` |

`deep-before/execute-notebook.py` equals the code cell of `checks/repair-a1/before-package/submission.ipynb`; `deep-after/execute-notebook.py` equals the current notebook's code cell (both asserted by `finalize_a1.py`). `deep-after-attempt1-partial/` is a prior worker's run that stopped after the first RESET (process exited; no result); it is retained and not used.

## Requalification on the repaired bytes (Python 3.12, site-packages origins, 32 pinned tycho_nooa sources verified)

| Check | Result | Evidence |
|---|---|---|
| New tests (`test_gate`, `test_local` incl. deep regression, `test_budget_accounting`, `test_deadlines`) | **49 passed** | `checks/repair-a1/new-tests-cp312.log` |
| Inherited runtime suite (16 files, copy of repaired package in `checks/repair-a1/test-root`) | **145 passed** | `checks/repair-a1/inherited-cp312.log` |
| Entrypoints / adapter | **37 passed** | `checks/repair-a1/entrypoints-cp312.log` |
| Gateway 110 games, normal | passed; 110 games, 550 Actor, 110 critic, 110-row parquet | `checks/repair-a1/gateway-110-cp312/` |
| Gateway reserve (Actor cap 1) | passed; 2 games, 2 Actor, 2 critic | `checks/repair-a1/gateway-reserve-cp312/` |
| Gateway critic-cap | passed; 2 games, 24 Actor, 8 critic (12/4 per game) | `checks/repair-a1/gateway-critic-cap-cp312/` |
| Gateway deep (package regression) | passed; 2 games, 11 Actor, 2 critic; 13 = 13 = 13; 0 deep requests/files; refusal observed | `checks/repair-a1/gateway-deep-cp312/` |
| `freeze_candidate.py --check` | `9431c419…` | `checks/repair-a1/runtime-freeze-check.log` |
| Launch gate | `HOLD: … final-review.json` (exit 2) | `checks/repair-a1/gate-hold.log` |
| `finalize_a1.py` reconciliation | exit 0 | `checks/repair-a1/finalize-a1.log`, `final-checks.json` |
| Bytes before vs after all tests | identical | `checks/repair-a1/snapshot-{before,after}.json`, `bytes-unchanged.txt` |
| Package hygiene | no `.pyc`/`__pycache__`/symlink; 72 files all manifested | `finalize_a1.py`, `launch_control.reconcile` |
| Tracked git status | equals `baseline-status.txt` | `finalize_a1.py` |

Total: 231 pytest passes, 0 failures; 4 gateway scenarios passed. Every gateway run materialized assets whose hashes equal the repaired `ASSETS.json` and whose `nooa-join/CHASSIS.json` equals the repaired contract, so all counts above are bound to the new bytes.

Why the 110-game fixture was rerun rather than reused: `finalize.py` and `audit_fixtures.py` bind each fixture's materialized assets to the current `ASSETS.json`; the T3 evidence binds only to the pre-repair `ASSETS.json` (`checks/repair-a1/prior-110-audit.json`: `materialized_assets_match` false against current, true against `before-package/ASSETS.json`). It is retained as supplemental evidence that the pre-repair ordinary path was consistent (550/110, 660 lifecycle admitted = completed, parquet 110 rows, no deep files) and cannot itself qualify the changed bytes. The rerun costs ~11 minutes locally.

## Hash identities

| Artifact | Pre-repair (T3/T4) | After repair (T3R) |
|---|---|---|
| `manifest.json` | `5cc3f11404566a29a0ccbc1b6fdcccca90c0154663e3e4eed5b0935a366fc70e` | `dcb716bc7809ccdc41322b25e48ab4a463b6bbe15ada1dc4b1886c3af05a2da8` |
| `EXPERIMENT-CARD.md` | `eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af` | unchanged |
| `submission.ipynb` | `910ac026855425732d7f9c8e5eabfdccd1607ff43d186d34c420902c4fe92183` | `3f38ad9b89d7f42423b9489b0ca2ee8a562c6993ed370f68364a98d43574e5a1` |
| `runtime/CHASSIS.json` contract | `03d0de3eebadb60f6ed62c88965f50f40beb7ffc9a643c0424e72a582b348d3f` | `9431c4196c221c4e92cfe5c275820b2f977140cab4171412318ba6166c5bafd1` |

Manifest sources: 84 → 85 (`finalize_a1.py` added to the freeze allowlist). Per-file before/after hashes for every changed source are in `checks/repair-a1/final-checks.json` → `changed_sources`; all 72 pre-repair package hashes are in `PRE-REPAIR-HASHES.json` and the pre-repair package bytes in `checks/repair-a1/before-package/` (verified equal). The card did not change: the scientific question, caps and identity are unchanged.

## Analysis-directory changes

- `qualify_gateway.py`: `deep` mode (regression fixture, above).
- `test_local.py`: `test_deep_refusal_official_gateway`.
- `freeze.py`: binds `finalize_a1.py`.
- `finalize_a1.py` (new): reconciles the repaired bytes against `checks/repair-a1/`, asserts the before/after deep evidence binding, the preserved pre-repair and independent-review hashes, all fixture invariants, and writes `checks/repair-a1/final-checks.json` and `CREATED-FILES.json`. It never rewrites `RESULT.md`.
- `PRE-REPAIR-HASHES.json` (prior worker): pre-repair manifest/card/notebook/package/independent-review hashes and tracked status.
- `REVIEW-REQUEST-A1.md`: fresh delta review request.
- `README.md`: pointer to this repair added at the top; body unchanged.
- `checks/repair-a1/commands.json`: exact argv/env of every command.

Not modified: `INDEPENDENT-REVIEW.md`, `independent-review/**`, `RESULT.md`, `REVIEW-REQUEST.md`, `finalize.py`, `checks/{gateway-*,*.log,...}` from T3. `RESULT.md` and `finalize.py` remain the T3 pre-repair record; `finalize.py` would now fail its asset-binding assertion by construction and is superseded by `finalize_a1.py` for the repaired bytes.

## Residuals

Unchanged from T3/T4 §9: GPU/container bootstrap and warmup, real model latency/capability, Kaggle's private parquet writer, the private 110-game inventory, provider timeout semantics, live one-push/submission, fresh quota, the structural absence of a remote public-mode path, and the inherited-suite dependence on dirty tracked files outside the manifest. Recommendation B (post-gameplay validation downgrade) remains open by instruction. An Actor that ignores the prompt and still calls `deep_investigate` wastes one cap-accounted Actor turn on the refusal; it cannot escape accounting.
