# Result: longer episode blocked by upstream rate limit

**Classification: `INTEGRATION_INVALID` — upstream HTTP 429 on the first request.** Exactly one authorized episode was launched with the new longer allowance. DekaLLM's shared upstream pool returned “qwen/qwen3.8-27b is temporarily rate-limited upstream.” The harness stopped; no retry, fallback, alternate endpoint/model, second episode, or deployment occurred.

## Actual live outcome

| Measure | Observed |
|---|---:|
| Request attempts / completed model responses | 1 / 0 |
| Inspections / candidate submissions | 0 / 0 |
| Completion allowance reserved for failed request | 4,096 tokens |
| Conservative prompt reservation for failed request | 11,433 tokens |
| Entire episode wall time | 4.128 seconds |
| Provider-reported prompt/completion tokens | **Unknown: no usage receipt** |
| Precise billed cost | **Unknown: no usage receipt** |
| Candidate / chosen work item / sealed evaluation | None / none / not run |

The request explicitly disabled reasoning, required tool use, disabled provider fallback and specified the same model. There is no returned model content with which to verify actual reasoning usage. Failed-request latency was **not independently recorded**: 4.128 seconds is episode wall time and an upper bound for its sole request, not an exact request-latency measurement. Completed-request latency logging exists but was never reached. This is a remaining instrumentation limitation; no live throughput or end-to-end speedup claim is supported.

The configured episode maximum was 16 requests, 16,384 actual completion tokens, 300,000 cumulative prompt tokens, 900 seconds, 10 inspections, four candidate submissions and $0.50. The single attempt reserved at most $0.052393 under the configured $1/M input and $10/M output routing ceiling. Public unauthenticated endpoint metadata is preserved in `provider-endpoints.json`; advertised rates were $0.15–0.45/M prompt and $2–3.20/M completion. Exact spend is unknown, **not zero**.

## Implemented and locally verified

- Actual provider completion usage is charged after each successful response; only the current request's `max_tokens` is reserved beforehand. Missing, malformed, negative, Boolean or over-reservation usage stops before another tool or request.
- Numeric pre-call telemetry covers remaining time, requests, prompt/completion tokens, inspection/submission attempts and conservative dollars. The loop reserves at least two requests and 4,096 completion tokens for commit/repair/evidence, switches early on low time or exhausted inspections, and removes/rejects inspections in that phase.
- Required tool calls avoid repeated prose-only rejection; two prose-only replies stop the loop. Recognized execution attempts count before argument parsing/execution.
- The initial prompt carries compact prefix facts and the exact incumbent, rather than duplicated full grids. An isolated inspection file exposes all chronological visible before/after grids, including intermediate states, without the suffix. Submission field types are explicit.
- Trusted observation, joint assessment, temporal ancestry and version-bound belief replay stay outside generated code. Each candidate ABI invocation remains a fresh bounded bubblewrap process with only current arguments. Output is capped while reading. No candidate runs in host.
- The guarded launcher parses only the authorized key entry and keeps it in same-process transport memory, not subprocess environments. Source/card/test hashes are verified against prelaunch and frozen in `STARTED.json`.
- Candidate freeze and a single postfreeze suffix-attempt ledger are implemented; the incumbent control result is reused from prelaunch instead of rerunning another control after freeze. Neither path was reached live.

## Measured performance before launch

Fixed visible evidence, identical incumbent adapter and inspection code; five sequential local samples per version. Raw samples and evidence/source hashes are in `PERFORMANCE.json`.

| Fixed-input measure | Previous implementation | V2 |
|---|---:|---:|
| Initial messages + tool schema, UTF-8 bytes | 14,582 | 6,645 |
| Median isolated inspection | 19.059 ms | 23.465 ms |
| Median visible candidate replay | 244.885 ms | 257.767 ms |

**Prompt payload shrank 54.43%. Local sandbox execution did not get faster in these measurements.** V2 inspection also serializes more complete chronological evidence; host scheduling and fixed order limit these small-sample comparisons. Prompt bytes are not tokenizer counts. The actual first request, including routing and telemetry, was 7,239 compact-JSON bytes versus the previous run's 14,678 bytes, but the previous run received model responses and this one did not.

The previous live episode took 27.142 seconds for four responses, 45,083 prompt / 961 completion tokens and $0.008152822, then exhausted requested allowances without submitting. This episode's 4.128-second failed attempt is **not evidence of faster construction**. Larger allowance was configured and exercised only up to the rate-limit blocker, not consumed.

## Verification and audit

`PRELAUNCH.json`: reference SHA-256 check passed; **28 reference tests passed; 23 local integration/adversarial tests passed**; unpaid dry run passed; all eight semantic checks passed. Controls verify incumbent visible support and suffix depth 2 with contradiction at historical step 8, plus a host-only witness supporting depth 3. The witness never enters construction evidence.

Frozen `audit.py` was executed and returned failure: it assumes completed-response fields for some checks and cannot fully reconcile an early HTTP abort. Its empty-response aggregate fields misleadingly sum to zero; **do not use `AUDIT.json` as a spend or usage report**. It remains unmodified for provenance. Post-run `audit_failure.py` independently reads the exact failed request, raw HTTP error, attempt ledger, prelaunch and frozen manifests: **all 16 failure-specific checks passed** (`FAILURE-AUDIT.json`). This establishes bounded abort, no retries, no construction/evaluation, and unchanged old/new frozen artifacts—not successful integration. No external call or evaluation is performed by either audit.

`RUNNER-RESULT.json` preserves the original runner result; `RESULT.json` adds explicit null usage/cost, failure details and performance reporting. No frozen source/card was edited after launch.

## Interpretation and next action

No candidate or work item was produced, and no suffix was evaluated for a model candidate. The rate-limit response is an infrastructure blocker, not model capability evidence. Historic LS20 steps 6–8 were already known to designers and withheld only from the constructor, not an untouched holdout. This remains a narrow singleton-commitment integration, not the full finite-domain prototype or game-general competence.

Per the precommitted card, stop and preserve the failed episode. A new live attempt needs a new experiment identity and separate authorization. Before any such attempt, improve failed-request timing and completed-versus-failed audit handling; do not rewrite this frozen run or silently retry it.

## Artifact location

`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-v2-20260913/`

Key artifacts: `EXPERIMENT-CARD.md`, `PRELAUNCH.json`, `STARTED.json`, `PERFORMANCE.json`, `RUNNER-RESULT.json`, `RESULT.json`, `FAILURE-AUDIT.json`, `audit_failure.py`, original `AUDIT.json`/`audit.py`, `raw/requests/00.json`, `raw/responses/00.error.body`, `raw/request_attempts.jsonl`, implementation modules and tests. Prior experiment and unrelated worktree files were preserved; no Kaggle/runtime changes were made.
