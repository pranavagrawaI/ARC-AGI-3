# V12 public1 guarded multi-action terminal result

## Verdict

**Kaggle version 10 ended `ERROR`. The run is a reliably source-bound negative interface-uptake result and an invalid deployment execution.**

Qwen did not submit `take_actions`. It committed two single actions, `RIGHT` then `LEFT`, and the third decision consumed the full 16,000-token ceiling as reasoning without emitting either required action tool. No batch was accepted, so the release guard was never exercised.

Claim ceiling: this run shows that merely exposing the guarded batch schema beside `take_action` did not produce an accepted batch on this `ls20` prefix. It does not show that batch release is safe or unsafe, does not establish eight-action reliability, and makes no gameplay-improvement claim.

## Remote identity and preservation

- Slug: `pranavagrawai/prototype-tycho-nooa-public-3-full-timing`
- Version: **10**
- Push started: `2026-09-02T10:57:04Z`
- First monitor status: `QUEUED` at `10:58:57Z`
- Later status: `RUNNING` at `11:09:30Z`
- Terminal status: `ERROR` at `11:39:34Z`, 2,550.135 seconds after push start
- Exact version-addressed output `/10` was downloaded before interpretation from `11:40:05Z` to `11:41:42Z`.
- Version-addressed source pull returned Kaggle's `403`; the current source was pulled immediately afterward from `11:41:42.715Z` to `11:41:44.112Z`.

The launch wrapper anomaly did not make the remote run ambiguous. The saved Kaggle CLI result is return code 0 and its stdout says `Kernel version 10 successfully pushed`; the monitor then observed the exact slug transition through `QUEUED`, `RUNNING`, and `ERROR`, and downloaded `/10`. The separately reported outer-wrapper exit 1 is therefore transport-wrapper evidence, not a failed-launch verdict.

Preserved bytes:

| Artifact | Files | Bytes | Canonical entries SHA-256 | Manifest SHA-256 |
|---|---:|---:|---|---|
| `kernel-v10-output-exact` | 184 | 30,988,044 | `2f916335f38c306317486fa9c17b50bbc9ef8554f87e1d4f805cc26cd6767d54` | `6e41e3a3639b34eadd5a44d13032177a61d4eab8231084c0afbc02e4b8c1ce3f` |
| `kernel-v10-source` | 2 | 416,111 | `9211aea61c5882ea87c1ce1f270811795dd31773c9def5391e42542ea2b3a773` | `3040221bd543727aa37f9110660359960ffbfe60377d51f142dddc670e2405ef` |

The immediate post-push source and immediate-terminal source are byte-for-byte identical:

- notebook raw SHA-256: `587bd74fb766b88e088ac7fe0ac0afe4766dc5470eac43cfd4aaee27a3e2df6b`
- metadata raw SHA-256: `caadb5cf148a018ef9fa2be382f0b172bda60c13666ffc92c61553702b27d391`

The remote notebook parses identically to the frozen local candidate. Its raw local hash is `194e781b9881bc752f97a0854f3f6a23fd9327f88f0a4a1c360c81669e8aae8e`; both canonical notebook objects hash to `acef1c2b20d7703ac503bc4f143a128fd15abfdec5b3301c052b49cfc0fe9bc0`. Remote metadata binds the private, offline, RTX PRO 6000 kernel to the competition, the accepted model/runtime datasets, and `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v9`.

The kernel log records wheelhouse revision 9, manifest `c24d0e1c1656fa6542b07a13ea1de4f6303efa4b5b5e078e2a5ec184ace67c8f`, 77 wheels/packages, `verified: true`, and installation of the Solver wheel. The frozen notebook would have raised before the launcher on an installed-source mismatch; reaching seven Actor calls is control-flow evidence that its packaged-source checks passed.

## Model requests and responses

Every persisted Actor request has a response and an `llm-trace` record. The server log contains eight successful `POST /v1/chat/completions` accesses: one warmup plus exactly seven Actor calls. There is no hidden-retry discrepancy.

All seven requests advertised the exact eight-action horizon and exposed both `take_action` and `take_actions`. Calls 1–5 were sequence 0 with `tool_choice=auto`; calls 6–7 were successors with `tool_choice=required` and only the two action tools. Every `take_actions` schema had `maxItems=4`.

| Call | Seq. | Used / remaining | Tool choice | Prompt | Completion | Reasoning | Finish | Emitted tool |
|---:|---:|---:|---|---:|---:|---:|---|---|
| 1 | 0 | 0 / 8 | auto | 13,909 | 6,365 | 6,077 | tool_calls | `run_python` |
| 2 | 0 | 0 / 8 | auto | 14,444 | 523 | 318 | tool_calls | `run_python` |
| 3 | 0 | 0 / 8 | auto | 15,201 | 6,274 | 6,167 | tool_calls | `run_python` |
| 4 | 0 | 0 / 8 | auto | 15,857 | 87 | 28 | tool_calls | `run_python` |
| 5 | 0 | 0 / 8 | auto | 16,328 | 5,277 | 4,676 | tool_calls | `take_action(RIGHT)` |
| 6 | 1 | 1 / 7 | required | 24,042 | 11,218 | 11,191 | tool_calls | `take_action(LEFT)` |
| 7 | 2 | 2 / 6 | required | 33,498 | 16,000 | 16,000 | **length** | **none** |

Totals: 133,279 prompt tokens, 45,744 completion tokens, 44,457 reasoning tokens, and 179,023 total Actor tokens.

The failing request SHA-256 is `497862da2f233256baede0fdc08a2487d1596c797212239b594486bcfe126ff4`; its response SHA-256 is `abe493595b86a0da141e6a8870f7350f3171806a13c0d016c8aa7d9388ba9e8b`. SGLang's exact terminal marker is:

> Required tool call missing from qwen3_coder output (0 chars)

The model did not simply miss that it had to act. Calls 6 and 7 each ended with an explicit final-model-call message stating that no later request existed, that only `take_action` and `take_actions` were available, and that the thread would fail without one valid action call. The response reasoning also discussed whether to queue actions. Call 6 deliberately selected one `LEFT`; call 7 reasoned toward a possible multi-action sequence but exhausted the ceiling before emitting it.

## Actions and batch trace

The two accepted Solver actions reconcile across the tool trace, durable workspace, official stdout, and official recording:

| Sequence | Solver | Official action | Recording time | State | Levels |
|---:|---|---|---|---|---:|
| 0 | `RIGHT` | `ACTION4` | `2026-09-02T11:19:41.965780Z` | `NOT_FINISHED` | 0 |
| 1 | `LEFT` | `ACTION3` | `2026-09-02T11:24:55.609382Z` | `NOT_FINISHED` | 0 |

The durable turns are `turn_001.json: RIGHT` and `turn_002.json: LEFT`; official stdout reports `ACTION4` and `ACTION3` at the same recording timestamps. The official recording has exactly two action-result rows under one game GUID.

Batch facts:

- accepted `take_actions` calls: **0**
- batch trace file: **absent**
- submissions: **0**
- submitted actions: **0**
- releases: **0**
- invalidations: **0**
- model calls avoided: **0**

Because no batch was submitted, neither successful continuation nor guard invalidation was tested. Seven Actor calls produced only two environment actions; the intended call-saving inequality went in the wrong direction.

## Causal classification

This was **not an allocator or model-server failure**:

- the pinned model verified and SGLang became ready;
- all seven Actor requests returned HTTP 200;
- the final response itself reports `finish_reason=length`, 16,000 completion/reasoning tokens, and no tool;
- no CUDA kernel exception marker appears;
- the exact SGLang marker names the missing required tool call.

The outer `failure_classification` string says `pre_request_failure:official_launcher:RuntimeError`, but that generic label is false if read literally: seven Actor requests completed. The discriminating evidence is the final checkpoint response and access log. The operational failure is **model noncommitment under the 16,000-token ceiling on sequence 2**, after the new tool was available but before any batch invocation.

The surfaced compatibility error says, “none was committed when `take_action` was the only tool left.” That wording is also stale for this run: the persisted successor request exposed `take_action` and `take_actions`. It does not change the checkpoint facts.

## Caps, gameplay, timing, and lifecycle

| Cap | Allowed | Observed |
|---|---:|---:|
| Public games | 1 (`ls20`) | 1 |
| Environment actions | 8 | 2 before failure |
| Actor calls | 13 | 7 |
| Successful chat accesses incl. warmup | 14 | 8 |
| Completion tokens per call | 16,000 | 16,000 max |
| Actor completion tokens | 208,000 | 45,744 |
| Launcher timeout | 7,200 s | no timeout |

Gameplay ended `NOT_FINISHED`, with zero completed levels and no win. No scorecard was finalized.

Timing facts:

- model-file verification: 237.970 s
- server model-API readiness: 142.023 s after launch
- warmup: 8.099 s; complete at 150.123 s
- official launcher: `11:11:37.561Z` to `11:33:01.794Z`
- recording creation to action 0: 479.902 s
- action 0 to action 1: 313.644 s
- action 1 to no-action failure: 486.185 s
- outer probe: 1,672.613 s

Containment passed. The owned SGLang server received SIGTERM, drained with zero remaining requests, and the lifecycle gate reported `term_only=true`, `quiet_window_valid=true`, empty post-run GPU-process inventory, and no listener on port 30000. The teardown classifier accepted the log. The official wrapper nevertheless returned effective code 71 because its failed agent path left a residual process group requiring termination; this did not prevent outer lifecycle closure.

## Competition boundary and next decision

This was a private offline batch run, not a competition rerun. Remote metadata has internet disabled; `system-before.json` records `competition_rerun=false`; the preserved operation ledger contains status/output/pull operations only. No competition submission command was issued, and no submission artifact was reached in the failed notebook path.

The frozen result map now applies: **revise the prompt/schema from these exact responses before another Kaggle run; do not auto-retry this candidate.** The result supports neither a three-game batch run nor a public-25 promotion. The next cheap test should specifically distinguish “the model notices a batch tool” from “the model commits it before open-ended reasoning consumes the call.”

Machine-readable facts: `TERMINAL-RECONCILIATION.json`.
