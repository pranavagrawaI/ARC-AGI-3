# T4R independent delta review — PASS

The exact repaired A1 bytes close T4 defect A and are locally launch-ready. This review is not launch authorization, competition-submission authorization, a remote result, or a capability claim. The original `INDEPENDENT-REVIEW.md` HOLD and `independent-review/**` remain untouched; this report addresses the newly bound candidate only.

Candidate: `deployment/kaggle/submission-20260917-actor-critic-competition-v1/`.
Run: `analysis/competition-submission-20260917-actor-critic-v1/`.
Target: `pranavagrawai/arc3-ac-competition-20260917-v1`.

## Exact identities

- Manifest SHA-256: `dcb716bc7809ccdc41322b25e48ab4a463b6bbe15ada1dc4b1886c3af05a2da8`
- Notebook SHA-256: `3f38ad9b89d7f42423b9489b0ca2ee8a562c6993ed370f68364a98d43574e5a1`
- Card SHA-256: `eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af`
- CHASSIS contract: `9431c4196c221c4e92cfe5c275820b2f977140cab4171412318ba6166c5bafd1`

`delta-review-a1/static-results.json` records independently recomputed bindings. All 85 manifest sources reconcile, all 72 package files are allowlisted, and there are no package symlinks or bytecode/cache files. The gate's actual in-memory notebook rebuild equals the notebook, all 13 embedded payload hashes reconcile to ASSETS, and the embedded archive's 24 entries include the exact CHASSIS and its 23 bound files. `freeze-check.log` records the independent runtime freeze check.

Metadata, title/SDK slug, command, identity, card and envelope are unchanged from the pre-repair manifest. Metadata remains private, internet disabled, RTX PRO 6000, competition attachment plus four datasets; the real gate executes the local SDK slug normalization. `budget-audit.json` recomputes the unchanged 32,400-second provider envelope, 110 games and caps 12 Actor/4 critic per game. Worst-case budget arithmetic remains inside the envelope; it does not establish private-provider timing behavior.

## Delta and executable boundary

`delta-review-a1/package-source.diff` contains the complete executable source delta:

1. `runtime/chassis/solver_agent.py::deep_investigate`: after argument validation, a nonempty `ARC_GAME_REQUEST_LEDGER` raises before the deep-call counter, evidence collection, direct `self._llm.acall`, or deep record publication.
2. `runtime/headless.py`: removes precisely the Actor system-context sentence advertising that method. The method's inherited docstring remains; an attempted use is still safely refused and costs a normal accounted Actor turn.

The only other changed package files are regenerated CHASSIS, ASSETS and notebook. Source comparison with all 72 saved pre-repair files confirms that `entry.py`, `play.py`, `budget.py`, shared LocalSolver, metadata, donors and policy/caps are unchanged. `qualification-source.diff` covers the deep fixture and freeze allowlist changes; `test_local.py` adds the matching regression and `finalize_a1.py` adds repair-specific reconciliation machinery. No source was removed from the manifest. Recommendation B was deliberately not applied.

`play.py:89` supplies one nonempty request-ledger path per game before constructing MyAgent. `runtime/shared_local_solver.py:63` forwards it to workers. A restarted/reset worker inherits the same game ledger; the cap-accounted Actor middleware still owns the call that attempts the refused tool. The guard precedes the direct deep request at `solver_agent.py:1794`.

The exercised chain is the exact notebook -> materialized entry/play -> official `Agent.main()` -> MyAgent adapter -> local_solver wrapper -> shared LocalSolver -> nooa_worker -> headless -> MdArcSolverAgent. Actor and critic remain live; no constructor integration or alternate execution path was added.

## Discriminating before/after reproduction

Both fresh runs use the unchanged original reviewer fixture `independent-review/qualify_gateway_deep.py`, the official Agent.main/REST gateway, synthetic games and a scripted local model. `delta-review-a1/reproduce_before.py` changes only the fixture PACKAGE pointer to the verified saved pre-repair package. It does not modify that package or fixture. Current-byte reproduction invokes the original fixture directly with `--games 2 --mode deep`.

| Fresh execution | Pre-repair | Repaired |
|---|---:|---:|
| Model server requests | 14 | 13 |
| Shared ledger / IPC model calls | 13 / 13 | 13 / 13 |
| Expected solver requests | 13 | 13 |
| Deep model requests / record files | 1 / 1 | 0 / 0 |
| Refusal visible in subsequent model requests | no | yes |
| Official completed games | 2 WIN | 2 WIN |
| Supervisor reconciliation | fails | passes |
| Result | reconciliation_failed | passed |

Evidence: `deep-before-reproduced/`, `deep-current/`, corresponding logs, and `deep-pair-audit.json`. The audit independently checks request bodies, ledgers, model-call rows, refusal, official outcomes and materialized asset hashes. Both materialized notebook code cells equal their respective exact notebook. The old deep call requests 32,768 output tokens and bypasses the ordinary count; the repaired trace has only Actor and 192-token critic requests. The package-owned deep regression also passed in the test suite.

The preserved T3R before-run defect was independently reconstructed as well. One initial reviewer-script assertion expected a shortened error phrase, whereas the real error prints a facts dictionary; the assertion was corrected to check its actual expected/observed fields. No candidate or historical evidence changed.

## Repeated tests and raw 110-game audit

All commands use `deployment/kaggle/.venv/bin/python` (CPython 3.12), bytecode disabled and fresh output/basetemp paths. Exact suite argv/env are in `delta-review-a1/test-commands.json`; `run_tests.py` executes them. Dependency guards resolve tycho_nooa/nooa/arc_agi/arcengine to site-packages and check 32 solver sources. The inherited runtime test copy is under `delta-review-a1/test-root/`.

- New manifest/payload/gate/budget/deadline/deep regression suite: **49 passed**.
- Inherited affected runtime/Actor-critic suite: **145 passed**.
- Entry point and adapter suite: **37 passed**.
- Total: **231 passed, zero failures**.

The repaired 110-game raw run was audited rather than rerun. `audit_fixtures.py` and `audit-fixtures.log` check current repaired assets, 110 unique completed WIN games, 550 Actor plus 110 critic requests, 660 admitted and completed lifecycle events with matching IDs, zero failed events, 110 worker sessions, per-game shared ledgers and IPC counts, and 110 complete parquet rows with no nulls/duplicate row IDs and matching terminal game coverage. Materialized CHASSIS equals the current repaired contract. Framework imports originate inside the materialized framework; dependencies are CPython 3.12 site-packages.

`audit_deep_pair.py` independently reruns the actual supervisor reconciliation over that 110-game launcher result and server count: 660 solver requests plus one fixture warmup, 661 expected/observed, verified. The warmup/server summary is a local fixture, not an actual SGLang warmup. Reserve and critic-cap repaired raw fixtures were also audited: 2/2 and 24/8 Actor/critic counts, respectively. The prior pre-repair 110-game run is not used to qualify repaired bytes.

## Gate proof and review publication

No real final-review.json existed at review start (`static-results.json`); neither real authorization nor consumption marker existed.

The rerun gate suite exercises missing/empty/malformed/list/duplicate review/auth, wrong bindings and identity, consumed markers, source/notebook/metadata drift, wrong command/run directory and other failures. The original 14 extra adversarial cases also all return HOLD with zero provider calls (`gate-extra.log`).

`gate_exact_review.py` tests the exact proposed final-review document against the real predicate and real immutable candidate/manifest, without retargeting or rehashing the manifest. Only review/auth reads and consumption/receipt writes are mapped to disposable files under `delta-review-a1/exact-authority-fixture/`; the provider command is intercepted. The proposed review is accepted. Ten further negative cases cover actual stale pre-repair manifest/notebook hashes and missing/malformed/empty authority files. All HOLD before provider invocation. An intercepted failing push creates the disposable consumed marker before the callback; a second push HOLDs. **Actual provider calls: zero.** No real authorization or CONSUMED file was written.

`final-review.json` is byte-identical to the tested `delta-review-a1/proposed-final-review.json`. Its status is uppercase `APPROVED`, because that is the real mandatory schema; lowercase `approved` is deliberately rejected by the gate. It includes `independent: true` and exact identity/target/manifest/card/notebook bindings. `final-review-sha256.txt` records its hash.

After publication, the real command is:

```
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B \
  analysis/competition-submission-20260917-actor-critic-v1/launch_control.py check \
  --run-dir /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/competition-submission-20260917-actor-critic-v1
```

Its required outcome is HOLD/exit 2 solely for absent authorization.json; the actual output and return code are retained in `delta-review-a1/real-gate-after-review.log` and `real-gate-after-review.json`.

## Preservation and limits

All 6,536 original review/HOLD files match PRE-REPAIR-HASHES. A broader independent starting snapshot covers 53,131 pre-existing run files; all remain byte-identical (`protected-before.json`, `preservation.json`). Candidate manifest sources are rechecked at completion. Tracked dirty status remains the original six modified files. Writes are confined to this report, delta-review-a1, and the newly issued final-review.json. No prior T3/T3R evidence was overwritten.

Unchanged residuals:

- Recommendation B remains open and non-blocking: post-gameplay output validation and inherited end-of-run contracts can turn completed gameplay into a notebook error. This review does not weaken them.
- CPU synthetic games, scripted model and local parquet callback do not qualify GPU/container bootstrap, real model latency/capability, Kaggle's private writer/identifiers or the private 110-game inventory.
- Provider timeout/queue semantics, fresh quota, model/data/framework attachments and exact-version retrieval require live preflight/observation. Refreshed official framework bytes can fail the immutable hash check.
- There is no remote public-mode gameplay path: commit mode writes dummy parquet. The first exact-byte remote gameplay is the scored rerun. The authorizer must explicitly resolve the prior remote-public-qualification prerequisite; this approval does not waive it.
- Inherited tests depend partly on unchanged dirty working-tree adapter/runtime/test files and historical raw evidence outside this candidate's manifest. Their pass is checkout-specific.
- A model still attempting deep_investigate spends a cap-accounted Actor turn on refusal. This is the selected A1 behavior, not an unaccounted request.

Next boundary: exact authorization plus live preflight before one push; competition submission requires authorization bound to the exact pushed version. No kernel push, competition submission, GPU operation, paid model call, provider write, or remote-success claim occurred in this review.
