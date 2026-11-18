# R10A-FROZEN-LAUNCH-REVIEW — APPROVED, awaiting Hermes release

Independent reviewer, 2026-09-16. **APPROVED for the exact frozen manifest below.** No blocking frozen-byte, authority, requested-source or gate finding remains. This completes the independent frozen launch review; it does not issue Hermes release, consume authorization, or perform a push. Actual remote attachment remains a required immediate post-push observation, not a fact that can exist before target creation.

## Exact identity and evidence binding

| Artifact | SHA-256 |
|---|---|
| Production manifest, 70 source files | `9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9` |
| Production authorization | `22cc840a91f394c061c8b9366ceb187fa982924039fef48328e486a6c5cdbb23` |
| Reviewed candidate inventory | `b2325fb93db5dfa16254a7d3d6c06b82a8b628bb61f4374276be6fe3e76e8aef` |
| Reviewed notebook | `87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3` |
| Approved card | `6425dc576370b0f68ea884ff5ea11cbf6989aa80bc08beaf0c86d3c8e4c72bc9` |
| Independent package decision | `3754a4535498a2908a6df75fc38789b80e327fee1fdea0ec9cbf7cc4e31834fc` |

Recomputed every manifest source hash independently, all 62 package hashes, and all 60 decoded notebook assets/EXPECTED hashes. Rechecked the prior package decision's report hash and input-map binding, every prior delta-review input, and every retained freeze receipt against EVIDENCE-HASHES. All match. `verify_frozen.py` returned 0 for the exact absolute run directory with `FROZEN_BYTES_VERIFIED_NOT_LAUNCH_APPROVAL`. Evidence: `verify-frozen.log`, `input-hashes.json`, `reconcile.py`, `reconcile.log`, `reconciliation.json`.

The package and gate bytes are the previously reviewed/reproduced bytes. Prior evidence established 16 package-wrapper passes, 141 gate passes and five reviewer gate-boundary passes. Those results are not rerun or relabeled as remote/model/GPU qualification here. The old HOLD/evidence and subsequent repair approval remain intact.

The card's “UNFROZEN” heading and README's preparation-stage “no freeze” statements are historical status prose within preserved artifacts. The later explicit freeze task and existing hash-bound manifest establish the present stage. No source edit or control reset is justified by those old labels.

## Requested target, resources and SDK mapping

- Exact target: `pranavagrawai/arc3-r10a-cd82-20260916`.
- CD82 pair, A then B; constructor only in B. The registry's additional game names do not authorize extra identities or runs.
- Private Python notebook; GPU on, TPU off; requested `NvidiaRtxPro6000`; notebook internet off.
- Competition requested: `arc-prize-2026-arc-agi-3`.
- Exactly four dataset sources:
  1. `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`
  2. `mahdiseddigh/kaggle-t4-stack-2026-08-28`
  3. `pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime`
  4. `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11`
- Kernel/model source arrays empty. Docker image request is the frozen metadata's digest-pinned image.

The installed CLI shebang selects `/home/pranav/.local/share/uv/tools/kaggle/bin/python`. Its SDK implementation hash matches the submitted evidence: `210e589ab1f7359ffb560aacfe1c003ac1bc844c366f4ff709bf3d13de4d5458` for `kaggle/api/kaggle_api_extended.py`.

Independently exercised the installed SDK's unchanged request-construction statements offline, using the actual typed `ApiSaveKernelRequest`, frozen metadata/notebook, timeout `3600`, and accelerator argument `NvidiaRtxPro6000`. A replacement client stops at request capture; network is forbidden by an audit hook. No real KaggleApi initialization/authentication or real client save occurred. `sdk_mapping_probe.py` and `sdk-mapping.json` show:

- `slug` equals the exact authorized target;
- `competition_sources` maps to `competition_data_sources` with the approved competition;
- `dataset_sources` maps to `dataset_data_sources` with all four approved sources;
- private/network/GPU/TPU flags and timeout/machine shape match;
- SDK notebook serialization only joins source lists, clears code outputs and serializes JSON. Parsed transformed notebook equals the corresponding transformation of the frozen notebook. The outgoing text hash is `f1611167564f284517b9fd00f037130103858e1ff3dfc99511a26cb475860793`; this standard transport representation is not the local manifest file hash.

This verifies the requested outgoing mapping, not server acceptance, mounted content, or actual attachment.

## Source access, quota and target receipts

Reviewed raw SDK-backed CLI receipts, exact argument arrays, timestamps, successful exit codes and empty stderr. Independently checked receipt hashes and parsed results. `source-access-summary.json` records:

| Source | Listed files | Evidence limit |
|---|---:|---|
| Competition | 148 | Read access, not attachment |
| Model dataset | 80 | Unversioned listing, not mounted hashes |
| Offline stack | 200 | More pages exist; not a complete inventory |
| SGLang runtime | 9 | Read access |
| Solver wheelhouse | 85 | Read access |

These observations were taken at approximately **2026-09-16 09:18:50 UTC**. The same receipt batch reports target search `Not found` and **26.39 GPU hours remaining**, exceeding the one-hour reserve. These are point-in-time observations, not guarantees at release. The frozen gate rechecks quota and target absence before consumption and revalidates local bindings afterward. No additional remote lookup was needed in this review; no credentials were read or printed.

## Numeric caps, cleanup and one-shot authority

The frozen card, manifest policy and active gate agree:

- One private push maximum; zero retries; zero competition submissions or dataset pushes; authorization `used=false`.
- Provider timeout 3600 s; work deadline 3100 s; guardian 3500 s with 8 s descendant cleanup; bootstrap reap 30 s. Startup ceiling 1200 s, launcher ceiling 1800 s. Local monitor expiry is explicitly not remote cancellation or proof of zero further billing.
- Each arm: 600 s admission, 200 executed actions, 129 incumbent calls. B's constructor shares its incumbent purse. Total 258 application calls plus one warmup =259 chats.
- Actor completion <=16384; critic <=192; reasoning off; normal HTTP <=90 s.
- B constructor: six requests, 240 s soft/245 s hard; POST <=30 s/client <=35 s; 4096/request, 24576 cumulative completion, 600000 cumulative prompt tokens, 10 inspections/four submissions; ordinary repair possible. Outer deadlines may stop earlier, as the card states.

Checked notebook guardian invocation, `donor_bootstrap.py:63,103,116-117`, `process_guard.py:29-49,55-89`, runtime cleanup `official_launcher_runtime.py:2442-2499`, and the already reviewed construction/relay/gate caps. Cleanup receipts and actual GPU/model identity remain run-time evidence requirements; no success is predicted from local fixtures.

`launch_control.py:153-201,255-288` requires the frozen review plus Hermes release, rechecks source/authority after read-only preflight, fsyncs exclusive CONSUMED before the sole push, and retains it after any uncertainty. The current user request confirms carried launch authority; the approved worker dispatch already limited it to one private push per sequential identity after review/Hermes gates. No new budget ruling is needed. R10b still cannot run until Hermes verifies collected R10a mechanical acceptance.

## Operational advisory: expected SDK warning, no retry

Installed SDK `kaggle_api_extended.py:6367-6377` deterministically prints a title/slug advisory: title `R10a CD82 constructor repair` slugifies to `r10a-cd82-constructor-repair`, while the requested slug is `arc3-r10a-cd82-20260916`. The offline captured request retains the exact authorized target. This advisory does not itself show a server rename or bad attachment.

The frozen gate deliberately treats any additional push stdout as uncertainty. If the provider returns an otherwise valid exact-target/version success receipt, the gate writes the exact `launch.json`, retains CONSUMED, then returns **2/HOLD** because of that advisory (`launch_control.py:278-288`). This is an expected read-only recovery branch, not grounds for another push. Tested the exact captured advisory through the real gate against an intercepted provider: `advisory-test.log`, **1 passed in 0.08 s**; exact version persisted, repeat push refused, read-only monitoring possible.

No gate behavior is waived by this approval. Hermes must inspect the retained receipt and exact version even if push returns HOLD. Source-invalid warnings or an unexpected target are not equivalent to this known advisory: preserve evidence, hold acceptance and escalate; do not retry, rename or create a replacement identity automatically. This bounded, already-tested uncertainty handling is why the advisory is not a launch-review blocker.

## Required immediate post-push readback

A remote attachment cannot be inspected before this new target exists. Immediately after the one authorized push (including the advisory/HOLD branch), Hermes/implementer must perform read-only SDK readback of the exact returned target/version:

1. Take version only from the preserved exact-target push receipt / `launch.json`; retain request and response evidence.
2. Use `ApiGetKernelRequest` with `user_name='pranavagrawai'`, `kernel_slug='arc3-r10a-cd82-20260916'`, and **`version_label=str(version)`**; call `get_kernel`. The installed generated type explicitly supports this field (`kagglesdk/kernels/types/kernels_api_service.py:589-645`). Do not rely on a latest-version default or assume CLI slug/version parsing sets the SDK version field.
3. Verify returned identity/version, privacy, network and accelerator metadata, competition attachment and all four dataset attachments. Resolve any server-returned version-qualified source IDs through read-only metadata and retain that mapping; do not silently discard mismatches. Record source/version/metadata evidence. Compare returned notebook content with the SDK-normalized reviewed content, allowing only known serialization/output differences.
4. Treat a temporarily unavailable field as unresolved pending read-only verification, not as established attachment. If attachment or identity is wrong, preserve receipts and hold scientific/mechanical acceptance; never remove CONSUMED or repush. This review grants no additional cancellation or mutation authority.
5. During the run, require mounted game/source/model/GPU identity and cleanup receipts. Read-only metadata cannot replace those checks. Collect only exact version/session paths and retain the final pending prediction distinction.

## Hermes final gate and release

Exact run directory:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-cd82-20260916`

Hermes should independently rerun from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/r10a-verification/verify_frozen.py --run-dir /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-cd82-20260916
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/r10-prep-20260916/r10a-verification/launch_control.py check --run-dir /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-cd82-20260916
```

Before Hermes release, the second command must HOLD for missing `hermes-release.json`. Only Hermes may write its release after independent checks. Required bindings (provided here as a schema example, not written by this reviewer):

```json
{"status":"RELEASED","authority":"Hermes","identity":"r10a-cd82-20260916","target":"pranavagrawai/arc3-r10a-cd82-20260916","manifest_sha256":"9310d597434699539041d342346e630f18ffa295d6f16a4a10205a2229134be9","notebook_sha256":"87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3","max_pushes":1,"retries":0}
```

Rerun `check` after release; expected GREEN_LOCAL_GATE still means provider preflight has not yet run. The authorized implementer must use the frozen one-shot gate, not bypass it. Changed frozen sources, manifest, approval bindings or authority invalidate this review; installed SDK drift requires mapping revalidation.

## Deliverables and scope

Production `launch-review.json` is independently authored with `status=APPROVED` bound to the exact manifest. Its extra evidence fields explicitly preserve the precreation attachment limit and required post-push readback. `schema-and-gate-check.json` records the actual schema match, successful frozen validation and expected full-gate HOLD for missing Hermes release. No release/consumption/launch record was written; no real remote/provider call occurred in this review. No frozen source, previous review, card, tracker or Git state was modified.

SUPERSET_WORKER_DONE
 task: R10A-FROZEN-LAUNCH-REVIEW
 summary: APPROVED exact frozen manifest; 70 hashes and prior approvals match; requested SDK mapping and source-access receipts verified. Actual attachment awaits required immediate post-push readback. Hermes release remains pending.
 files: analysis/r10-prep-20260916/independent-launch-review/; analysis/r10-prep-20260916/r10a-cd82-20260916/launch-review.json
 checks: Read-only verify_frozen PASS; independent hash/embedding/review/receipt reconciliation; offline typed SDK mapping; exact advisory no-retry recovery probe; actual review schema and expected missing-Hermes HOLD.
 handoff: Hermes independently reruns exact commands and releases the manifest. One-shot gate only; expected SDK advisory may cause post-admission HOLD. Read back exact version/attachments immediately; never retry. No push performed here.
