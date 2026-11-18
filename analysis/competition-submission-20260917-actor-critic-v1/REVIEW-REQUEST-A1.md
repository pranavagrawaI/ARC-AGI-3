# Delta review request after repair A1 — no approval granted

Supersedes `REVIEW-REQUEST.md` for the current bytes. This is not a request to launch. Neither `authorization.json` nor `final-review.json` exists; the gate is HOLD / exit 2 for that reason only. The T4 review (`INDEPENDENT-REVIEW.md`, `independent-review/`) is preserved byte-for-byte (hashes in `PRE-REPAIR-HASHES.json`, re-asserted by `finalize_a1.py`).

Identity: `pranavagrawai/arc3-ac-competition-20260917-v1` (unchanged). Any fresh `final-review.json` must bind:

- `manifest_sha256` `dcb716bc7809ccdc41322b25e48ab4a463b6bbe15ada1dc4b1886c3af05a2da8` (was `5cc3f114…`)
- `card_sha256` `eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af` (unchanged)
- `notebook_sha256` `3f38ad9b89d7f42423b9489b0ca2ee8a562c6993ed370f68364a98d43574e5a1` (was `910ac026…`)

## Changed hashes (everything else in the manifest is byte-identical to the T4-reviewed bytes)

| Source | Before | After |
|---|---|---|
| `…/runtime/chassis/solver_agent.py` | `7f7cead434bbabde5e66d3ea7669a363684be20a203f84a456586aefb26c67f2` | `c20879bee0db6cf58a94d3cf95474bae50a6f8d56e012d17c0c8b7950094641c` |
| `…/runtime/headless.py` | `72f123b4ed410e574ad6bdafe890e075ec0926450ecf7cdbcc3c2568082a2674` | `3c9c3e75ab6c6f96aa6ddfa3df83cbd8ea68f8c0985ff0f881cb2e1ff355dcc4` |
| `…/runtime/CHASSIS.json` (contract `03d0de3e…` → `9431c419…`) | `eb3c366698d6f1753f45d3aa33550205635a650233ee870591eb892f2ec546a8` | `c1971355520d5eeeb581e086f6dab9fba3f664b26b928355de0238880314e4e8` |
| `…/ASSETS.json` (`nooa-join.zip` payload only) | `35962b71ac4f862bd5d23b208055a50f33adf4715ad559bafe4ef0d32d38ab60` | `886edf071b9ceec420653494bc389d780d84c67c9abdb7c920ecde9449570f6e` |
| `…/submission.ipynb` | `910ac026855425732d7f9c8e5eabfdccd1607ff43d186d34c420902c4fe92183` | `3f38ad9b89d7f42423b9489b0ca2ee8a562c6993ed370f68364a98d43574e5a1` |
| `qualify_gateway.py` (adds `--mode deep`) | `d99b4152fcf4da5481004f1d89b63fb6ecbe1466b0b1cf8114f44a3ad68bd3dc` | `5ceb3dc66219e7576931d750ad7fa42b6179ba9805b29b454f501091aa53969d` |
| `test_local.py` (adds `test_deep_refusal_official_gateway`) | `a595640a1ecdd043d76b6187c85c4d02f9eecd072dea8e1a79b3bb1894b36c1a` | `22bc564f34dce10a04340f062f007c4c34e5d7a0eabb2060cd76291bdd992cea` |
| `freeze.py` (binds `finalize_a1.py`) | `82b5bc6bd3e7d4e1fa365300e01c77c0627a123bf2c0910563aa748d4ab0d773` | `c643e16f03197a5d45c3e6f244f6785174bd48ce6fd29883b42323b3d786a732` |
| `finalize_a1.py` (new) | — | `a4cd45248a66641da3c7ebb1505e8d7f65dd36cc494e9491adf0055cb69042a9` |

`…` = `deployment/kaggle/submission-20260917-actor-critic-competition-v1`. Source diffs: `diff -r checks/repair-a1/before-package <package>` (five files) and `diff checks/repair-a1/before-current-artifacts/qualify_gateway.py qualify_gateway.py`.

## Exact checks to repeat (all commands with env in `checks/repair-a1/commands.json`; use fresh output paths)

1. Reviewer fixture, repaired bytes: `independent-review/qualify_gateway_deep.py --games 2 --mode deep` → expect `RESULT.json.status == passed`, `server_requests_total == shared_ledger_rows == expected_launcher_solver_requests == 13`, `deep_investigation_records == 0`, `deep_max_tokens_requested == []`. Compare with `checks/repair-a1/deep-before` (14 vs 13, 1 record, reconciliation failed) and confirm `deep-before/execute-notebook.py` equals the pre-repair notebook code cell.
2. Package regression: `qualify_gateway.py --games 2 --mode deep` → `DEEP-SUMMARY.json` with `refusal_observed true`, `deep_model_requests 0`, `deep_investigation_files 0`, 13 = 13 = 13; and `test_local.py::test_deep_refusal_official_gateway` inside the 49-test new suite.
3. Ordinary paths on the new bytes: `qualify_gateway.py --games 110 --mode normal` (110/550/110), `--mode reserve` (2/2/2), `--mode critic-cap` (2/24/8); in every output verify `notebook/<asset>` hashes == `ASSETS.json` and `notebook/working/nooa-join/CHASSIS.json` == `runtime/CHASSIS.json` (contract `9431c419…`), so the counts bind to the repaired bytes. T3's `checks/gateway-*-cp312` bind to the old assets (`checks/repair-a1/prior-110-audit.json`) and are supplemental only.
4. Suites: inherited 145 (from a copy of the repaired package, `checks/repair-a1/test-root`), entrypoints/adapter 37, new 49; `freeze_candidate.py --check`; `launch_control.py check` → HOLD on absent `final-review.json`; `finalize_a1.py` exit 0; `independent-review/snapshot.py` before/after identical.
5. Static: the guard is the first statement after argument validation in `deep_investigate` and precedes `maximum = …`; `'Use await self.deep_investigate'` is absent from `runtime/headless.py` and nothing else in that prompt changed; `entry.py`, `play.py`, `budget.py`, `kernel-metadata.json`, `DONORS.json`, `shared_local_solver.py` are byte-identical to T4.

## Discriminating questions for this delta

1. Is `ARC_GAME_REQUEST_LEDGER` guaranteed set in every competition worker (it is one of the five keys forwarded by `WORKER_ENVIRONMENT_KEYS` and set per game by `play.py`), so the guard cannot be bypassed by a worker restart or reset?
2. Does the wasted Actor turn on a refused call remain inside the 12-call cap and the admission deadline (it is admitted by `_guard` like any Actor call), and is that acceptable given the advertisement is gone?
3. Are the T4 §9 residuals and the open recommendation B still the only remaining boundaries, i.e. does nothing in this delta widen the remote-unexercised surface?
4. T4 §9 item 9 (no remote public-mode path; the first remote execution is the scored rerun) is unchanged and is the authorizer's decision, not this request's.

Limits are those of T3/T4: CPU-only synthetic games, scripted model, local parquet callback, cp312 site-packages origins. No remote or scientific claim is made. Do not self-author approval from these results.
