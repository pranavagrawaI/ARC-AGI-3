# Exit 74 diagnosis and local repair

The official launcher returned **0**. The surrounding `nooa_public_probe.py` returned **74**, deliberately, after its session validator rejected the successor's chassis name. This is not interpreter finalization and does not require launcher exit instrumentation.

Exact path in the deployed source:

- `nooa_public_probe.py:183` requires the literal `q6-stall-authority`.
- Actual worker metadata reports `q6-summary-boundary-20260906`, matching the actual frozen CHASSIS.json identity and contract `24c291bc...`.
- `_summarize_session` raises `chassis pin differs for ls20`.
- The outer exception handler stores that reason in `solver_sessions_error` and assigns `effective_returncode = 74` at line 472.
- `official-launcher-result.json` preserves both facts: `launcher_returncode: 0`, `returncode: 74`. The generic outer error text calls the probe subprocess the launcher, which obscured the distinction.

This was introduced when the successor manifest was renamed while the inherited validator retained its donor identity. Earlier local fixture tests did not validate the actual manifest-to-worker identity relationship.

## Repair and evidence

`exit-review/nooa_public_probe.py` is an isolated corrected source file. It reads the expected identity and contract from CHASSIS.json under NOOA_JOIN_ROOT, and requires worker metadata and launch provenance to agree with that frozen contract. It does not accept arbitrary identities or remove pin checking.

Six local regression tests pass (`TESTS.txt`): actual frozen-source failure; corrected real-session acceptance; incorrect identity rejection; mutually agreeing but wrong worker/launch contracts rejected against the manifest; missing manifest rejection; and identity changes governed by the manifest rather than another hardcoded name.

`replay.py` verifies the downloaded payload against its manifest and replays session validation against retained logs. A temporary directory restores the empty workspace/ls20 marker omitted from Kaggle downloads; raw files are read only. Existing packaged request reconciliation passes with 155 expected/observed solver requests and one warmup, 156 total. Stop evidence confirms 100 actions; there is no ABORT, integrity failure, or recorded reasoning. `REPLAY.json` retains the results.

The original run and frozen notebook remain ERROR and unchanged. These are post-hoc corrected-validator checks, not a new kernel run or silent rewriting of qualification. The corrected source is ready for integration into a future candidate; no replacement package was frozen or launched during this diagnosis.

## Consequence for solver work

The two completed summarizer calls consumed 111.14 seconds. The launcher reports 3005.16 seconds of gameplay, so their summed model wall time is about 3.7% of gameplay wall time (background scheduling means this is not a measured counterfactual saving). Their semantic quality and the value of eviction remain separate questions. No GPU rerun is needed just to discover the cause of 74.
