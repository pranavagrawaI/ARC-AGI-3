# SUPERSET_WORKER_BLOCKED — R10A-PUSH / recovery and collection

## Observation

The frozen gate was executed exactly once. Provider accepted version **1** at **pranavagrawai/r10a-cd82-constructor-repair**, different from the authorized target **pranavagrawai/arc3-r10a-cd82-20260916**. Gate exit 2 / HOLD: push URL target/version mismatch. CONSUMED.json remains intact; launch.json was not created or fabricated. No repeat push, rename, source/control patch, cancellation, submission or R10b run.

Supplemental current status reached **COMPLETE**. Receipt: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/push-20260916/monitor/TERMINAL-OBSERVATION.json`. Terminal current metadata still reports currentVersionNumber=1. Explicit SDK version_label="1" get_kernel, status and output-list calls continue to return 404. Intended-target explicit lookup returned 403. All failed requests and responses are retained separately from production controls.

## Readback evidence and limits

Current-only metadata verifies private=true, enableInternet=false, enableGpu=true, enableTpu=false, requested NvidiaRtxPro6000 and Docker digest, competition attachment and all four dataset attachments. Parsed returned notebook equals the reviewed SDK-normalized notebook in initial supplemental mapping. Terminal comparison is recorded in terminal-summary.json. Missing kernelDataSources/modelDataSources are unresolved even in raw response; no empty arrays were inferred.

Current metadata id=134588849 and currentVersionNumber=1 do not establish an explicit script-version lookup. Raw current lastRunTime predates this push and is retained without interpreting it as execution timing. Latest/current evidence is not relabeled version-bound.

Exact-session collection is **BLOCKED**: the explicit version-1 output endpoint returned 404 again after supplemental COMPLETE. Zero session artifacts downloaded. collection-plan.json lists allowed fixed receipts and the frozen collect_plan.py derivation; no broad bundle or credential-bearing logs collected. Runtime/game/model mount identities, constructor exercise, scoring and cleanup are unverified.

Terminal CLI quota reports **25.92h GPU remaining** (4.08h used /30h). Raw SDK quota is also retained, including malformed duration and total discrepancy. Neither certifies exact run cost; no cancellation or zero-reservation guarantee inferred from it.

## Commands, paths, hashes

- Exact one-shot command argv/stdout/stderr/exit: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/push-20260916/gate-push-receipt.json`.
- Raw provider receipt: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-cd82-20260916/push-56e324f3e2004081979fd2f2a9c40e4f.json`.
- Production manifest SHA-256: `9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9`; all 70 frozen source hashes still match.
- CONSUMED SHA-256: `ac8c4f7190fdc3aa47109b336f5498c580534fa5ccb59abc8f8601f6315fe542`.
- Supplemental request/raw response: supplemental-current-raw-request.json and supplemental-current-raw-response.json.
- Notebook/attachment comparison: supplemental-readback-comparison.json; terminal-summary.json.
- Explicit requests and raw nonsecret responses/errors: returned-*.json, intended-*.json, recovery-*.json, terminal-readback-*.json, output-list-*.json, monitor/*.json.
- Recovery command sources: readback.py, recovery-probe.py, raw-current-readback.py, recovery-monitor.py, output-probe.py, terminal-readback.py. Executed with `/home/pranav/.local/share/uv/tools/kaggle/bin/python -B`; monitor used `-u`. These are observation-only recovery scripts, not frozen-gate replacements.
- Quota receipts: terminal-quota.json and terminal-quota-cli.json.
- Evidence byte inventory: EVIDENCE-HASHES.json.

## Handoff

Mechanical/scientific acceptance remains **HOLD** for actual identity mismatch and unresolved exact-version provenance/collection. PM/Hermes must resolve provider version-addressing semantics or supply an independently bound session-output mapping before exact-session collection can complete. Do not retry the push or fabricate production launch state. Supplemental terminal observation does not repair either blocker. No further remote mutation is authorized by this recovery report.
