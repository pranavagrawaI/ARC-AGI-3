# Independent final launch review — APPROVED

Exact manifest: `0f01cc456dc452a1895dcd30e8594a7c9deee55e0f90c4cc8c227295d84b8aab`.
Run: `analysis/actor-critic-construction-kaggle-20260914T174335Z`.
Scope: final coordinator gate, card, authorization, exact package binding, metadata and read-only monitor. The previously approved local package was not reopened or edited. Unrelated working-tree changes are outside this review.

## Evidence and decision

- All **106 manifest sources** independently rehashed with zero mismatches, including gate `3432379ab7cfa74c738edaa8613f7b6ce758b3f11a10308165fe81bb2ed0010b`, gate tests `6cc2650fffc1ea2123a391dc2c06fa464b588420272b8c3d33988b5648cb55c2`, package manifest `234a62ec3727d6593043324f3f471877755da0fc146dfaef6d754375531213b3`, notebook `0fc78e34ff45d82e95d815bcd8b39ddef7c6c8067cd2775446df50dea3053962`, and card `013926a414bcaa8c54440e96c92e1ffde19f984f8cd4d7bee4943ab4c784903c`.
- Frozen gate suite independently executed: **119 passed**, zero failed (0.27 seconds in retained run). Provider subprocesses intercepted by recorder; disposable fixtures only. Coverage includes malformed/missing/empty controls, wrong types and bindings, negative reviews, source drift, live-preflight drift/revocation, quota and absence rejection, competing consumption, uncertain push, warning recovery, exact-version SDK adapter, transient monitoring and local deadlines.
- Exact real run first exercised with only the prospective review-file load substituted, no provider calls or run mutations. Rechecked source hashes, unused authorization, absence of CONSUMED.json and launch.json immediately before approval.
- Wrote the requested exact `launch-review.json` exclusively. **Immediately afterward, the unmodified real gate `main(['check','--run-dir', EXACT_ABSOLUTE_RUN])` read that actual file and returned exit 0 / GREEN_LOCAL_GATE / provider_checks NOT_RUN.** The provider recorder remained empty. Approval readback matched exactly and consumption remained absent.

## Launch path and controls inspected

`launch_control.py` is the coordinator execution path, not the older packaging-only `launch_once.py` (which always refuses launch). `push` checks exact absolute run identity, complete package/source coverage, all hashes, strict metadata, affirmative exact-manifest review and total unused authorization before any provider call. It checks displayed GPU quota and successful exact absence, revalidates controls and manifest, then exclusively creates and fsyncs CONSUMED.json (including parent directory) before its sole push. The exact command requests timeout 3600 and NvidiaRtxPro6000. Consumption survives errors, uncertain admission, output parse failure and warnings; no retry is available. Returned positive version and exact target URL are bound to the saved push receipt. Warning-bearing success retains the launch identity for read-only recovery but returns HOLD.

The card's target and exact notebook title both resolve to `pranavagrawai/actor-critic-construction-kaggle-smoke`; installed official slugify verified the title. Frozen metadata explicitly selects private, GPU true, TPU false, internet false, exact four datasets, competition `arc-prize-2026-arc-agi-3`, no kernel/model sources, and pinned image `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`. Gate checks the attachment values directly; title and image are also protected by the reviewed metadata digest. Installed CLI source confirms notebook serialization, exact attachment projection, docker_image forwarding, title slug warning and session_timeout_seconds forwarding. Server readback and actual device evidence remain mandatory after admission.

The monitor validates the returned launch/consumption/receipt binding and runs an SDK status request with explicit version_label in the installed Kaggle interpreter. The actual adapter is exercised by the offline suite. It handles COMPLETE, ERROR, CANCELLED (SDK CANCEL_ACKNOWLEDGED normalized), retries only read-only observations, and preserves failed receipts. Local deadline reports remote_stop_established false; no cancellation, push, dataset publication or competition submission path exists in monitor. The default 3600-second monitor is within the card's initial maximum 7200; coordinator must not choose a larger first-monitor duration although the generic argument allows up to 86400 for read-only monitoring.

## Prospective limits and residual uncertainty

The card explicitly separates 3600 seconds of requested provider runtime from the one-GPU-hour reservation, 3500-second notebook guardian plus 8-second retirement, 3100-second work cap, 1200-second server startup, 900-second admission, 24 total server chat requests including warmup, 23 application calls, 20 actor/summarizer calls and 8 executed actions. Conditional permission covers one push with zero retry, not a comparative-performance claim. The card accurately describes the test Agent driver (not Agent.main) and assimilation as unscored.

Initial retained successful receipts report 28.47h GPU remaining and Not found; these are historical preflight evidence, not refreshed observations by this reviewer. The real push gate must perform its live quota/absence checks. Displayed quota granularity, queue delay, timeout enforcement latency, target installation/GPU/model behavior and actual billed usage remain unqualified remotely. No exact-session cancellation mapping is qualified or promised. Local monitor exit is not remote shutdown. Output/source retrieval and final quota audit remain coordinator-owned read-only follow-up, not functionality claimed to exist in this gate. Unknown admission must never be repaired by resubmission.

## Artifacts

- `source-verification.json`: execution receipt with every expected/actual source hash and prospective gate result.
- `gate-tests.json`: retained offline suite receipt.
- `exact-real-gate.json`: approval creation and immediate real-gate validation receipt; zero provider calls, consumption absent.
- `verify_review.py`: repeatable read-only reconciliation / prospective predicate check.
- `finalize_review.py`: one-time exclusive approval writer and immediate real-gate check; deliberately refuses an existing approval.
- Run `launch-review.json`: authoritative exact-manifest affirmative review.

No blocking finding. No launch or real provider/model request made. No candidate, manifest, card, authorization, gate or protected source was edited. Only this review directory and the authorized run approval artifact were written.
