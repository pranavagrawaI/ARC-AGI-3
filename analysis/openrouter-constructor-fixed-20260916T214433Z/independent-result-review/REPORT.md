# Independent fixed-pair result audit — DONE

## Conclusion

**Both freezes are supported as timely promoted, source-bound visible ABI artifacts. Neither is a validated game model.** Both arms submitted a correctly nested work_item and an actual JSON array of source lines on their first submission. The recorded sealed evaluator completed initialize → predict → assimilate over the one visible RIGHT transition. No failed submission or repair cycle occurred in this pair.

The reasoning contrast, initial parity, receipts and costs reconcile. **Provider-reported pair cost: $0.049969794**, zero unresolved exposure. Candidate inventory and prior-artifact bindings remain unchanged. No API/auth access, paid call, experiment rerun, candidate execution or candidate edit was performed during this audit.

## Recomputed observations

| Measure | Fixed off | Fixed on |
|---|---:|---:|
| Requests / HTTP 200 receipts | 5 | 4 |
| Prompt tokens | 114279 | 86596 |
| Completion tokens, including reasoning | 912 | 3912 |
| Reasoning tokens | 0 | 2898 |
| Successful inspections | 4 | 3 |
| Submissions / failed submissions | 1 / 0 | 1 / 0 |
| Accepted ABI freeze | yes | yes |
| Episode seconds | 32.516013 | 84.508214 |
| Provider-reported charged USD | 0.023971450 | 0.025998344 |

All nine raw response usages equal their receipts and ledger entries; raw request/response hashes match. Costs independently recompute exactly using uncached prompt $0.214/M, reported cached prompt $0.15/M, and completion $2.55/M. Completion already includes reasoning. Every reservation matches exact outbound bytes plus framing and 131072 advertised output exposure; each known-cost-plus-reservation fits $2. Reported output fits each requested 4096 cap; cumulative prompt/output and request caps fit. Known charges are not reservations, and this is provider-reported accounting rather than an independent account-statement audit.

Every off response reports zero reasoning tokens with no reasoning payload. On reports **1140, 11, 1731, 16**, with reasoning payload on each response. Exact outgoing configuration is disabled versus enabled/medium, exclude=false in both. Routing metadata identifies direct Reka selection, attempt 1, canonical qwen/qwen3.8-27b-20260814, no transformation pipeline; outgoing route remains reka/fp8 with no fallbacks. This establishes provider-reported treatment, not hidden compute or checkpoint identity.

Complete first requests are equal after removing only reasoning; shared SHA-256 is `78b1575ca9361084dacac80f8bd66c95a6f87d9f4997301a793e6ef98e05889d`. The initial hash also equals the prior pair's: submit_candidate is unavailable until inspection succeeds, so its changed nested schema is not present in the initial request. Subsequent actual submission requests contain the fixed schema. The treatment trajectories then diverge naturally. All subsequent feedback and assistant tool/reasoning continuation messages reconcile with previous raw responses and events.

## Submission and freeze chain

Paths below are relative to the fixed candidate directory. The exact submissions are `live/off/child/raw/005.response.raw` and `live/on/child/raw/004.response.raw`, `/choices/0/message/tool_calls/0/function/arguments`.

Both work_item values are real JSON objects with exactly the nine required fields and typed arrays. Off uses decision `d`, obstacle `o`, claim `c`, remaining_scope `r`, dependencies `["d"]`; on uses uppercase equivalents. Both declare operation=ground, reasoning_need=does_not_block_decision, evidence_refs=[1], completion_checks=[1]. They pass the structural validator, but these single-letter descriptions/dependency names carry negligible explanatory content. The fixed interface did not establish a substantive scientific claim or dependency chain.

Both candidate_source values are actual arrays of Python source-line strings. The host's declared `"\n".join(lines)+"\n"` conversion reproduces the progress source and both child/root freeze sources exactly:

- Off source SHA-256: `f21b6413295bf114804eae769526fae6bcb47c8608533f25b08cef7ddc7364d2`
- On source SHA-256: `05926d38fd80b74176f64d4c71a592d071cf1bbb7d7c69aeb50d491da5a209fa`

Raw tool calls equal tool-events; progress transcripts contain the raw parsed responses unchanged. Recorded submission feedback is evaluated/passed rather than rejected. No local adapter corruption or silent string-to-object repair appears in that chain. This does not independently exclude provider-side serialization or assign authorship to checkpoint internals.

Root FROZEN.json is byte-identical to child/FROZEN.json in each arm, including source digest and ABI-only claim. Root results equal child results plus the root authority field. Supervisors report exit 0, reaped=true, hard_timeout=false, at 32.641140 and 84.628809 seconds. Ledgers are receipted/stopped=false with zero unknown exposure. These satisfy the reviewed parent promotion conditions, rather than relying on child-file existence alone. Both roots correctly retain supported_model=false and prospective=not_run.

## What the sealed evaluator actually checked

`engine/host.py:31` installs evaluate_visible_abi through bound_constructor. This is the generic evaluator in `engine/evaluate_visible.py`, **not** the semantic observer/replay helpers also present in integration_boundary.py.

The generic path checks initialize returns a dict; predict returns exactly one dict branch with state and dictionary commitments (optional declared_dependencies must be a list); assimilate returns a dict after receiving the predicted successor, actual next grid, and exactly `{ "status": "unscored" }`. It compares no commitment against a measured game target. Required coverage is empty, static dependency completeness false, suffix not_run.

`engine/integration_boundary.py::SealedBoundary._invoke` executes each method in a fresh qualified process, validates rootfs before/after, and requires completed exit 0, valid startup seal, JSON stdout and bounded JSON result. The retained evaluator records passed visible replay and three ordered RPC entries. I independently reconstructed all three argument digests in each arm from the exact prefix and the short inspected source semantics, without executing the candidate: initial before-grid, predict state plus RIGHT, and predicted successor plus after-grid and unscored verdict all match recorded hashes.

Evidence limit: successful raw per-method stdout/startup seals are not separately retained here; the transcript/evaluation records are runtime testimony supported by bound evaluator/boundary source, source digests and matching argument digests. This audit did not rerun sealed execution or obtain an independent OS attestation.

## Vacuity and model behavior

Both predict functions copy the current grid and increment step; neither models the observed spatial transition. Off assimilate retains its predicted grid and ignores observation. On assimilate copies the newly observed grid, updating memory only after observation. That difference is visible in source, not evidence of predictive accuracy.

The commitment dictionaries are syntactically nonempty but **vacuous as measured predictions**:

- Off: note = “opaque grid shift, not yet characterized”.
- On: assumption = “directional dynamics unmodeled with one sample”.

No corresponding targets are required or scored. No completed level, gameplay result, prospective query, semantic support or dynamics efficacy is established. Calling these “successful ABI freezes” is accurate; calling them accurate, supported or useful game models would exceed the evidence.

## Descriptive comparison to the old pair

The earlier pair used 6 requests per arm, produced 4 off / 2 on work-item rejections, and froze neither source; cost was $0.069457910. The fixed pair used 5 off / 4 on requests and reached ABI evaluation/freeze on both first submissions; cost was $0.049969794. It used fewer requests and lower reported total cost, while on again used more reasoning/output and more wall time than off.

This supports the predeclared observation that the fixed pair crossed the previously failing interface boundary. It does **not** isolate schema from feedback, provider serialization, stochastic generation or serving/time variation. No rejection feedback was exercised in the fixed live pair, so it provides no live repair-success measurement. Both conditions froze; this singleton pair shows no demonstrated reasoning advantage on the binary freeze outcome. It cannot estimate treatment efficacy or checkpoint-only causation.

## Bindings and offline reproduction

Reviewed inventory remains `bba4d707c6c5c6edae7e818ca68ed8b0c969066832509b4acd2500f29b16276a`; every inventoried file matches. All **1195** snapshotted prior-experiment files, **70** frozen R10 sources, and hashed fixed-candidate independent-review artifacts match. INPUT-HASHES.json binds all read inputs; CHECKS.json contains per-call calculations, sources, evaluation and comparison. Inputs were rechecked for drift before completion.

From repository root:

```bash
python3 -B analysis/openrouter-constructor-fixed-20260916T214433Z/independent-result-review/audit.py
```

The audit only reads local evidence and writes its checks/hash manifest under independent-result-review/. It neither imports nor executes candidate code. EVIDENCE-HASHES.json binds this report and audit artifacts.

Next permitted decision is review of these descriptive results. No new run, implementation, acceptance of model dynamics, Kaggle launch or R10b clearance follows. Existing R10 HOLD remains unchanged.

SUPERSET_WORKER_DONE task=OR-CONSTRUCTOR-FIXED-RESULT-REVIEW; summary=both timely source-bound ABI freezes reconciled, semantic vacuity explicit; files=REPORT.md,audit.py,CHECKS.json,INPUT-HASHES.json,EVIDENCE-HASHES.json; checks=offline raw/transcript/receipt/ledger/parity/RPC-digest/preservation reconciliation; handoff=descriptive result only, no causal or efficacy claim, no new authority.
