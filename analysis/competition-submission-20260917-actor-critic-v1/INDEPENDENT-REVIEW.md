# T4 independent exact-byte review — `pranavagrawai/arc3-ac-competition-20260917-v1`

**Verdict: HOLD.** No `final-review.json` is issued.

One reproduced defect blocks: the Actor-callable `deep_investigate` tool makes a model request that bypasses the per-game Actor cap, the shared request ledger and the lifecycle records, and a single such call anywhere in the 110-game run makes the end-of-run whole-model request reconciliation raise after all games and the scorecard close, turning a complete rerun into a failed notebook. It is the only reasoning-on request path in the package (max_tokens 32768, `enable_thinking: true`), it is advertised in the Actor's system prompt, and no fixture can observe it because the scripted Actor never calls it. Everything else T3 claimed was reproduced exactly on the cp312 site-packages environment and is recorded below so the next freeze only needs to re-run, not re-review.

Reviewer: independent of T3. Read-only on the candidate; all artifacts under `independent-review/`. No kernel push, competition submit, GPU, paid model call or provider write occurred. Candidate, manifest, card, notebook, tests and launch tooling bytes are unchanged (`independent-review/bytes-unchanged.txt`).

Workspace: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus`. All commands below ran from that directory with `deployment/kaggle/.venv/bin/python` (CPython 3.12.14; `tycho_nooa`, `nooa`, `arc_agi`, `arcengine` all resolve to `.venv/lib/python3.12/site-packages`).

## 1. Hashes and manifest reconciliation — PASS

```
sha256sum analysis/competition-submission-20260917-actor-critic-v1/manifest.json  -> 5cc3f11404566a29a0ccbc1b6fdcccca90c0154663e3e4eed5b0935a366fc70e
sha256sum analysis/competition-submission-20260917-actor-critic-v1/EXPERIMENT-CARD.md -> eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af
sha256sum deployment/kaggle/submission-20260917-actor-critic-competition-v1/submission.ipynb -> 910ac026855425732d7f9c8e5eabfdccd1607ff43d186d34c420902c4fe92183
```

All three equal the claimed values. Independent recomputation (`independent-review/snapshot.py`): 84 manifest sources, 0 mismatches; 72 package files, all manifested, no unmanifested file, no symlink, no `__pycache__`/`.pyc`; 12 manifest-only sources are the run-dir gate/card/tests/scripts. Manifest `run_dir`, `package_dir` and `command` are exact absolute strings for this worktree; `metadata` equals `kernel-metadata.json`. `launch_control.reconcile` (run through `check`) passed every binding before reaching the absent review file: `HOLD: [Errno 2] No such file or directory: .../final-review.json`, exit 2 (`independent-review/gate-check-prereview.log`).

Embedded notebook: 2 cells (markdown `identity`, code `entry`), outputs empty, 13 base64 payloads = 651 754 bytes, names and SHA-256 equal `ASSETS.json` (recomputed by `test_local.py::test_payloads` and by the gate's in-memory rebuild `json.loads(submission.ipynb)==build.notebook()`). `nooa-join.zip`: 24 stored entries (23 `CHASSIS.json` files + `CHASSIS.json`), all timestamps 2026-09-17 00:00:00, compress type 0 — deterministic.

Provenance of files without a DONORS binding: `install.py` is v15 notebook cell 2 with one change (`shutil.rmtree(join_root)` → `raise RuntimeError('Join destination exists')`); `official-hashes.json` equals `runtime/build_notebook.py::OFFICIAL_SOURCE_HASHES` (v11, remotely matched by the v15 run on 2026-09-12). Runtime diffs from the v11 donor are confined to `chassis/solver_agent.py` (shared-ledger/cap guard, lines 885–1082), `shared_local_solver.py` (+4 `WORKER_ENVIRONMENT_KEYS`), `nooa_public_probe.py` (cap-based budget-stop acceptance), `CHASSIS.json` (rehashed) and 8 test files. `play.py` diffs from v15 are the shared ledger, critic deadline, loopback-only gateway override, 110-game check and import-origin guard.

Bytes after every review run: `diff snapshot-before.json snapshot-after.json` → identical.

## 2. Packaged execution path — PASS (Actor and critic both on the production path)

Static trace, confirmed by fixture evidence:

`submission.ipynb` (rerun branch writes 13 payloads to `/tmp`, `import entry; entry.main(root)`) → `entry.py` (budget.validate, work deadline, fixture-override refusal, `install.py` subprocess ≤1800 s, official-hash check of the pinned mount, framework copy, `official_launcher_runtime.py` = `exact_request_outer_runtime.py` fa7a1519…, byte-identical to the v15 remote run) → outer runtime starts SGLang, warms up, then runs `--official-launcher-probe submission_play.py` with `/tmp/solver-venv/bin/python`, `--output <out>/official-launcher-result.json --timeout-s <launcher-60>` → `play.py` imports `agents.templates.my_agent` (= `runtime/adapter.py`, 9c2fb321…, byte-identical to v11), `MyAgent(Agent).main()` (official `agents/agent.py` 85655f7a…) → `choose_action` → `local_solver.create_solver` (wrapper = v11 `local_solver.py`) → `runtime.local_solver` (= `shared_local_solver.py`) `LocalSolver` → `nooa_worker.py` → `headless.py` → `MdArcSolverAgent` (`chassis/solver_agent.py`) whose `_guard` middleware (line 870) admits every Actor/summarizer call and, on trigger, performs the critic call at line 981 (`self._llm.acall(..., tools=None, max_tokens=192)`) using `epistemic_critic` (imported at line 45) and `objection_ledger`.

Fixture evidence (T3's cp312 runs, audited by `independent-review/audit_fixtures.py`, output `audit-fixtures.json`): `import-origins.json` places `agents.agent`, `agents.templates.my_agent`, `runtime.local_solver`, `local_solver` inside the materialized `working/ARC-AGI-3-Agents`; `worker-metadata.json` records `agent: nooa.MdArcSolverAgent`, `chassis: submission-runtime-20260912-v11`, `contract_sha256 03d0de3e…`, command `<venv python> <materialized nooa-join>/headless.py`; critic requests reach the model server with no `tools` and `max_tokens 192`; `request_lifecycle.jsonl` carries `request_purpose: critic` with triggers `level_advanced`, `reserved_budget_boundary`, `eight_actor_calls`.

## 3. Competition mode — PASS with residuals

- Default rerun: `entry.py` production branch uses `Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')` and `/kaggle/working`; `play.py` defaults to `http://gateway:8001`, `OperationMode.ONLINE`, `arc_api_key='test-key-123'`, all supplied games, and refuses `len(games)!=110` outside the fixture. Any `ARC_FIXTURE_*`, `SUBMISSION_GATEWAY_BASE_URL` or `NOOA_AGENT_COMMAND` in a production environment raises `Fixture override in production` before `install.py`; a non-loopback gateway override raises in `play.py` even in fixture mode (`independent-review/production-branch-probe.json`, static section all true; the dynamic probe stops at `/kaggle` creation on this host, so ordering is established by reading `entry.py` lines 33–40: override check at 37–38 precedes the `install.py` subprocess at 40).
- Commit/non-rerun: the notebook's `else` branch and `entry.main` write exactly `[['1_0','1',True,1]]` with columns `row_id, game_id, end_of_game, score` and nothing else (`test_local.py::test_commit_exact_notebook` — materialized dir contains only `working/`). No bootstrap, no model.
- Output contract: `entry.validate_output` requires the four columns, no nulls/duplicate row_ids, game set equal to `game-progress.jsonl`, and a terminal row per game. The 110-game fixture parquet: 110 rows, 110 unique games, dtypes `str/str/bool/float64`, `end_of_game` all true. **Residual:** the writer in the fixture is T3's callback, not Kaggle's sidecar; and a post-hoc `validate_output` failure raises *after* the gateway already wrote the parquet, which can only downgrade a complete run to a notebook error (see §10, fix B).
- Metadata: `competition_sources=["arc-prize-2026-arc-agi-3"]`, `is_private=true`, `enable_gpu=true`, `machine_shape=NvidiaRtxPro6000`, `enable_internet=false`, `enable_tpu=false`, four dataset sources and the pinned `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…` image — identical to the live-verified R9 cd82 metadata except `id/title/code_file` (`donor-metadata.json` diff).
- Slug: `/home/pranav/.local/share/uv/tools/kaggle/bin/python -c "from slugify import slugify; print(slugify('arc3-ac-competition-20260917-v1'))"` → `arc3-ac-competition-20260917-v1` == metadata id suffix; the installed SDK 2.2.4 `kernels_push` derives the slug exactly this way (`kaggle_api_extended.py:6368`) and only warns on mismatch.

## 4. Budget audit — PASS arithmetically; one cap leak (§10)

`independent-review/budget_audit.py` → `budget-audit.json`, using the package's own formulas (`budget.admission`, `entry.py` `remaining`, `outer.bounded_launcher_timeout`, `play.py` per-game reserve):

| bootstrap | startup | entry remaining | launcher eff. | play timeout | first admission | games end | work alarm | entry kill | provider |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 300 | 28020 | 28020 | 27960 | 27.55 s | 28320 | 31020 | 31200 | 32400 |
| 1800 | 1200 | 27990 | 27990 | 27930 | 27.27 s | 30930 | 31020 | 31200 | 32400 |

Identity `110·(admission + 2·90 + 15 + 30) + 180 == play timeout` holds (`test_local.py::test_budget`). Worst case leaves 90 s to the work alarm, 180 s outer cleanup, then 1200 s (= 300 final output + 900 uncertainty) to the 9 h cap. v15 remote measurements: bootstrap+startup ≈ 360 s, orderly shutdown 12.1 s, outer overhead beyond the launcher 337.5 s — inside the 1800/1200/180 allocations. Impossible envelopes raise `ValueError` before model startup (`entry.py` line 74) and again in `play.py` before the scorecard opens.

Per-game bounds: admission deadline absolute from `game_started`; critic deadline = admission + 90; worker `ARC_GAME_WORK_DEADLINE` = admission + 195 (absolute, `shared_local_solver.decide` shrinks the watchdog to it); `close()` worst case 3 × 10 s = the 30 s cleanup reserve. Rolling admission recomputes from remaining time and games, so overheads shrink later windows rather than exceed the cap; it only crashes if cumulative unreserved overhead exceeds ≈3000 s.

Shared cap: `_guard` reads `ARC_GAME_REQUEST_LEDGER` (one file per game index, survives worker restarts/resets), counts non-critic rows (Actor **and summarizer**) against `ARC_COMPETITION_ACTOR_CAP=12` and critic rows against `ARC_COMPETITION_CRITIC_CAP=4`; a critic is admitted at most once per guard call and never when `critic_spent >= critic_cap`. Reserve: when the Actor is exhausted by time or count, `actor_spent > 0` and `critic_spent == 0`, one `reserved_budget_boundary` critic runs before `stop_budget()`. All five env keys are forwarded by `WORKER_ENVIRONMENT_KEYS`. Reproduced: reserve scenario 2 games → 2 Actor / 2 critic, both stops `request_cap_exhausted: true`, `actor_calls 1`; critic-cap scenario → 24 Actor / 8 critic, per game 12/4, ledgers max 12/4 (`independent-review/gateway-reserve-cp312/RESULT.json`, `gateway-critic-cap-cp312/RESULT.json`).

**Cap leak:** `MdArcSolverAgent.deep_investigate` (`runtime/chassis/solver_agent.py:1772–1817`) calls `self._llm.acall` directly (line 1792) with `max_tokens=int(ARC_DEEP_MAX_TOKENS or 32768)` and `enable_thinking: true`, outside `_guard`: no deadline check, no ledger row, no `request_lifecycle`/`model_calls` row (it writes `deep_investigations.jsonl`). `ARC_DEEP_MAX_CALLS` defaults to 1 per worker session and is not in `WORKER_ENVIRONMENT_KEYS`, so `entry.py` cannot lower it. The Actor prompt advertises it (`runtime/headless.py:262`). See §10.

## 5. Local reproduction on cp312 — PASS (counts match T3)

| Suite | Command (abridged; full argv mirrors `checks/commands.json` with new basetemp) | Result |
|---|---|---|
| New tests | `PYTHONDONTWRITEBYTECODE=1 ARC_DISABLE_DOTENV=1 PYTHONPATH=<run>/checks/local-deps-cp312 .venv/bin/python -m pytest test_gate.py test_local.py test_budget_accounting.py test_deadlines.py -q -p no:cacheprovider --basetemp=independent-review/new-tests-tmp` | **48 passed** in 2.85 s |
| Inherited runtime | same interpreter over the 16 runtime `test_*.py` files in `independent-review/test-root/…/runtime` (copy verified equal to manifest, 72/72), with T3's `analysis`, `deployment/kaggle/{agent,runtime,tests}` symlinks | **145 passed** in 49.7 s |
| Entrypoints/adapter | `SUBMISSION_TEST_RUNTIME=<test-root runtime> … pytest test_entrypoints.py test_adapter.py` | **37 passed** in 164.7 s |
| Gateway reserve | `qualify_gateway.py --games 2 --mode reserve --output independent-review/gateway-reserve-cp312` | passed: 2 games, 2 Actor, 2 critic, supervisor 1+4 closed, perturbed counts rejected, workers reaped |
| Gateway critic-cap | `… --mode critic-cap …` | passed: 2 games, 24 Actor, 8 critic |
| Gateway 110 (T3 artifacts) | `independent-review/audit_fixtures.py` | 110 unique complete games (all `WIN`, 2 levels), 550 Actor + 110 critic on the server, ledgers 5/1 per game, `model_calls` 550/110/0 summarizer, lifecycle 660 admitted = 660 completed, 0 failed, 110 worker sessions, parquet 110 rows / 110 games / 0 nulls, scorecard 110 environments, supervisor 661 = 661, dependency origins site-packages, 32 solver sources verified, no traceback in stderr |

Total: 230 passed, 0 failed — equal to T3. The 110-game run was not repeated: its raw artifacts are internally consistent on every count I could derive independently.

Note on the inherited 145: 13 of those tests (in `test_lifecycle`, `test_reasoning`, `test_contingent_worker`, …) resolve `ROOT=HERE.parents[3]` and import `deployment/kaggle/tests/test_my_agent.py` plus the working-tree `deployment/kaggle/{agent,runtime}` (tracked, *modified*, not in the manifest), and two (`test_prediction_feedback::test_replay_real_band_contradiction`, `test_evidence_interface::test_actual_transition_exposes_existing_before_and_exact_diff`) read raw evidence under `analysis/deployment-probes/`. Without T3's `deployment/kaggle/{agent,runtime,tests}` and `analysis` symlinks in the test root they fail (13 and then 2 failures, reproduced before adding the symlinks). Their pass count is therefore a property of this checkout, not of the package bytes alone.

## 6. Fail-closed launch gate — PASS

`test_gate.py`: 20 review/authorization negatives (missing, empty, malformed, list, duplicate key, wrong identity/target/manifest/card/notebook hash) × 2 files, 10 other negatives (consumed, title, metadata, payload, source, extra file, manifest identity, command, run-dir slash, `used`), positive validate without consumption, push consumes before provider and never retries — all with the provider call intercepted. Extra negatives in a disposable exact-bound copy (`independent-review/gate_extra_negatives.py` → `gate-extra-negatives.json`): `independent:false`, lowercase status, uppercase manifest hash, `max_pushes:2`, `timeout_seconds:3600`, truncated command, trailing-slash `run_dir`, `used:0` (int, not bool), wrong `package_dir`, metadata without competition source, manifest re-indented after review, package symlink, package `.pyc`, relative `--run-dir` → **14/14 HOLD**, 0 provider calls; restored fixture → `READY`.

Mechanics confirmed by reading `launch_control.py`: `reconcile` requires `str(run)==str(RUN)` and `RUN.resolve()==RUN`; `equal()` checks type and value per key; `authorization.json`/`final-review.json`/`CONSUMED.json` may not be manifest sources; `push()` creates `CONSUMED.json` with `O_CREAT|O_EXCL`, fsyncs file and directory, then runs `COMMAND` (`kaggle kernels push -p <pkg> --timeout 32400 --accelerator NvidiaRtxPro6000`, exact absolute strings) once with `timeout=180` and writes `push-receipt.json` with `open('x')`. Review/authorization are outside the manifest and bind `identity`, `target`, `manifest_sha256`, `card_sha256`, `notebook_sha256`.

## 7. Packaging hygiene — PASS with a note

No credential patterns, no `.env`, no dotfiles, no symlinks, no bytecode, no files >200 KB other than the 873 KB notebook; embedded payloads decoded and scanned: no `/home/pranav`, keys or PEM blocks. Five inherited v11 probe files under `runtime/` (`openrouter_probe.py`, `typed_self_test_live_probe.py`, `heterogeneous_self_test_probe.py`, `procedural_transfer_probe.py`, `procedural_transfer_reliability_probe.py`) contain the string `/home/pranav/projects/arc-agi-3/.env` as a lookup path (no secret); they are in `CHASSIS.json` `sources`, not `files`, so they are **not** embedded in the notebook and `kernels push` uploads only `code_file` and metadata. Dead weight, not leakage.

## 8. No pre-existing artifacts modified — PASS on available evidence

`git status --porcelain=v1 -uno` equals `baseline-status.txt` (`independent-review/tracked-status-check.txt`); the six tracked modified files have mtimes from 2026-09-03…09-09. Files with mtimes in the T3 window (14:44–15:34 IST) outside the two authorized directories are all under `analysis/world-model-utility-retry-20260917T091830Z/` — a separate concurrent workstream (directory created 05:49 IST, referenced by no T3 source). No T3 write outside the two directories was found.

## 9. Remote-unexercised classification (not local defects)

1. GPU/container bootstrap (`install.py`) and SGLang warmup on RTX PRO 6000 — bytes remotely proven only via v15 (2026-09-12), with the one-line `rmtree`→`raise` change.
2. Real model latency: first-game admission ≈27.5 s means ≈1–4 Actor calls plus one critic per game at observed 7–40 s/call; completion is feasible, scoring potential is near zero. Feasibility only, per the card.
3. Kaggle's private gateway writer and its exact `game_id`/`row_id` values vs `validate_output`.
4. Private 110-game inventory (the `len(games)!=110` guard is fatal if it differs).
5. Provider timeout/queue semantics and whether `--timeout 32400` carries into the rerun (T2 open item).
6. Official-hash binding includes `ARC-AGI-3-Agents/.git/logs/HEAD`; a refreshed competition attachment fails closed during the rerun and consumes the daily submission.
7. Whole-run fail-closed policies inherited from probe runtime: any single failed/timed-out model request (`request_failure.json` → `_model_evidence` raise), any worker retirement failure (`_MODEL_GENERATION_RETIREMENT_FAILED`), and the end-of-run server request-count contract, all abort the run *after* gameplay rather than degrade. Observed remote base rate with thinking off: 0 failures in ≈1 200 admitted requests; the one observed remote timeout was a reasoning-on request.
8. Live one-push, exact-version retrieval, fresh quota (25.92 h snapshot must be refreshed).
9. T2's "private public-mode qualification run before any submit": this package has **no remote public-mode path** (non-rerun = dummy parquet only), so exact-byte remote public qualification is structurally impossible; the first remote execution of `entry.py`/`play.py`/the ledger guard will be the scored rerun. The authorizer must either waive that prerequisite explicitly in `authorization.json`-adjacent evidence or accept a different-bytes public probe as a proxy. This is a decision, not something T4 can discharge.

## 10. Defects and minimal fixes

### A (blocking) — `deep_investigate` escapes the cap and breaks the request contract

Reproduction (exact package bytes, notebook materialized, official `Agent.main`, official REST gateway; only the scripted model reply differs from T3's fixture so that the first Actor turn executes `r = await self.deep_investigate("Which rule moves the piece?", ["decision_packet"])`):

```
PYTHONDONTWRITEBYTECODE=1 ARC_DISABLE_DOTENV=1 PYTHONPATH=<run>/checks/local-deps-cp312 \
  deployment/kaggle/.venv/bin/python -B independent-review/qualify_gateway_deep.py --games 2 --mode deep \
  --output independent-review/gateway-deep-cp312
```

Result (`independent-review/gateway-deep-cp312/RESULT.json`, `model-requests.jsonl`, `DEEP-SUMMARY.json`, `RECONCILE-ERROR.txt`):

| Observed | Value |
|---|---|
| Both games played to `WIN`, scorecard closed | yes |
| Requests seen by the model server | 14 |
| Shared ledger rows (Actor 11 + critic 2) | 13 |
| `expected_launcher_solver_requests` from persisted evidence | 13 |
| Extra request | request #1: no tools, `max_tokens 32768`, `chat_template_kwargs.enable_thinking true`, `reasoning.enabled true` |
| `deep_investigations.jsonl` records | 1 (`deep-1`) |
| `reconcile_launcher_request_count` | `RuntimeError: official-launcher request-count contract did not close: … expected 13, observed 14` |

In production that `RuntimeError` is raised by `official_launcher_runtime.py` after the launcher exits, `result.json` becomes `failed`, `entry.py`'s `subprocess.run(check=True)` raises, and the notebook errors after a complete rerun. If the deep request instead times out at the 90 s client limit (likely for a 32 768-token thinking request on a 27B model), the failure is the same (`observed_successful_total != observed_total`). Either way the request is not charged to the 12-call Actor cap.

Minimal fix (pick one; both require rebuild → freeze → re-test → re-review):

- A1 — `deployment/kaggle/submission-20260917-actor-critic-competition-v1/runtime/chassis/solver_agent.py`, insert before line 1786 (`maximum = int(os.environ.get("ARC_DEEP_MAX_CALLS", "1"))`):
  `if os.environ.get('ARC_GAME_REQUEST_LEDGER'): raise RuntimeError('deep investigation is disabled under competition request accounting')`
  and delete the sentence at `runtime/headless.py:262–263` (`'off. Use await self.deep_investigate(...) only for a hard abstraction failure whose ' 'answer could enable many actions, never for routine inspection. '`) so no Actor call is wasted discovering the refusal. Update `runtime/CHASSIS.json` (`files` hashes for `chassis/solver_agent.py`, `headless.py`, `contract_sha256` per `runtime/freeze_candidate.py:29`), then `python3 -B build.py`, `python3 -B freeze.py`.
- A2 — without touching the chassis: `entry.py` line 59 `os.environ.update` block add `'ARC_DEEP_MAX_CALLS':'0'`, and `runtime/shared_local_solver.py:58` add `"ARC_DEEP_MAX_CALLS"` to `WORKER_ENVIRONMENT_KEYS`; `maximum=0` makes line 1787 raise before any model call. Prompt still advertises the tool, so a wasted (cap-accounted) Actor call remains possible.

Regression evidence required at the next review: rerun `independent-review/qualify_gateway_deep.py --mode deep` and show `server_requests_total == shared_ledger_rows`, no `deep_investigations.jsonl`, `RESULT.json.status == passed`; add the equivalent assertion to `test_local.py` or `qualify_gateway.py`.

### B (recommended, non-blocking) — post-gameplay validation can only hurt

`entry.py:90` `validate_output(...)` and the inherited end-of-run contracts raise after the gateway parquet exists. Recommend writing a validation record (`qualification/output-validation.json`) and returning normally in production instead of raising; keep the raise in fixture mode. This does not change the card's invalidators (they are evaluated post hoc from the record) but stops a schema guess about the private writer from producing "notebook error" on top of a complete run.

## 11. Residual limitations of this review

- CPU-only, scripted model, synthetic games, local parquet callback; the production `official_launcher_runtime.py` → SGLang → `submission_play.py` path was not executed end to end here or by T3 (its arguments were read and match the v15 remote invocation shape).
- The inherited 145-test count depends on dirty tracked repo files and raw probe evidence outside the manifest (see §5).
- I did not repeat the 110-game fixture; T3's raw artifacts were audited programmatically and found consistent.
- No remote or scientific claim is made or implied.

Review artifacts: `analysis/competition-submission-20260917-actor-critic-v1/INDEPENDENT-REVIEW.md` and `independent-review/` (scripts, logs, fixture outputs, hash snapshots). Nothing else was written.
