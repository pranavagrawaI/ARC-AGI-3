# Exact-version submission containment, 2026-10-21

Implemented locally; no launch, submission, upload, queue/timer/incumbent change, staging, or commit.

`submit_draw.py` validates all of `quarantine.json` before importing Kaggle or refreshing credentials. The checked-in policy quarantines only `pranavagrawai/m2-f1-screen1-f1` v1 and `pranavagrawai/m2-f1-screen2-f1x` v1. Kernel case and numeric version spelling cannot bypass a match. A new version, unrelated kernel, or valid empty policy remains eligible under the existing rules.

Policy entries require a canonical kernel, positive integer version, nonempty reason and evidence list. Missing, unreadable, invalid JSON or malformed entries return 1, log the hold, and contact no Kaggle boundary. A matched identity returns 5 and logs its reason. Exit 5 differs from prerequisite failure/pending (3/4), so `daily_draw.py` does not treat quarantine as permission to fall back.

With the existing Oct 22 queue and observed D11/D12 CSV, the merge still returns 3 because D12 scored zero. The existing scheduler then tries F1x v1; that second invocation returns 5 before Kaggle imports or submission. This deliberately permits the day to remain unused until an eligible candidate is explicitly selected. Direct, queued, and empty-queue F1/F1x v1 submissions hit the same guard. No scheduler file was edited.

Evidence basis: existing D9/D12 receipts identify exact v1 artifacts and submission refs. `analysis/f1x-hidden-zero-diagnosis-20261021/submissions-listing.json` records COMPLETE 0.00 for refs 57342990 and 57477659. D10 ref 57385881 scored 30.88 with the same F1x scriptVersionId as D12. The diagnosis FINDINGS hypotheses are not treated as root-cause proof; this change only contains reuse of observed failed artifacts.

## Pre-mortem and alternatives

- A daily-only guard would leave direct calls unprotected: place the guard at the shared submission boundary.
- A family blacklist would reject qualified successors: match exact kernel/version and test v2 and unrelated controls.
- A missing/malformed file or invalid later entry could silently permit submission: validate the complete file, fail closed, test invalid schemas/fields and an earlier match followed by an invalid entry.
- Reusing exit 3 would authorize fallback: use exit 5 and exercise the real daily-to-submit chain with Kaggle mocked.
- The guard could disturb duplicate-draw or prerequisite behavior: retain the existing code and verify duplicate suppression and prerequisite exit codes through main.

## Verification

`python3 -B -m unittest analysis/hidden-draws-20261010/test_quarantine.py analysis/hidden-draws-20261010/test_requires.py`

24 tests pass: 13 existing and 11 new. External SDK imports, credentials, API and CLI calls are mocked and observed; healthy-control assertions require an actual submission command and receipt, while blocked cases require zero external interactions and no receipt/quota artifact. Tests use temporary directories and a fixed Oct 22 clock. The integration tests retain real `daily_draw.main` and `submit_draw.main`; only the Python process boundary is bridged in process for that chain.

Red/green logs preserve the original missing quarantine behavior, missing/malformed-policy behavior and case-alias bypass, followed by their passing checks. `submit_draw.before.py` preserves the pre-edit source. `git diff --check` passes.

Parent integration verification also ran the actual checked-in CLI against F1x v1 and observed exit 5; `actual-cli.json` preserves its command, output and checks. This is a local guard check, not a live Kaggle submission test.

## Bounded risk measurement and review

`python3 -B analysis/hidden-draws-20261010/quarantine-20261021/measure_risk.py before|after|expanded`

No coverage/radon packages are installed. The measurement reuses the repository's existing AST/stdlib-trace estimator from `solvers/m2_planning_tools/measure_risk.py`; no dependency, threshold or CI gate was added. Only submit_draw.py is measured. Before/after use exactly the unchanged 13-test suite: main CC 9 to 11, measured coverage 0 to 0, CRAP 90 to 132. This exposes the old suite's absence of main coverage; it is not a claimed risk improvement. The new loader, unexecuted by those old tests, scores 306 in the old-suite run.

The separately labelled expanded 24-test universe gives main CC 11, statement coverage 94.44%, CRAP 11.02075; new `load_quarantine` CC/CRAP 17 at full statement coverage; `quarantine_reason` CC/CRAP 3 at full statement coverage. Expanded maximum is 17. These expanded results are not compared as an improvement against the old-suite baseline. There is no threshold or violation count. The extra loader branches express required schema validation; no extraction or exclusions were used to lower the estimate.

Reviewed the tracked diff, new test/policy files, call chain and raw evidence. No supported correctness or scope findings remain. Live Kaggle behavior and the unresolved hidden-run zero cause remain untested. Existing scheduler race/idempotency limitations outside this guard were not redesigned. A valid empty policy intentionally removes quarantine; policy edits therefore remain an operational choice.
