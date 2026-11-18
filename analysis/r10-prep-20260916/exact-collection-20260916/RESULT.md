# R10A exact collection and descriptive reconciliation

SUPERSET_WORKER_BLOCKED
 task: R10A-EXACT-COLLECTION
 summary: Numeric-version collection and descriptive reconciliation completed with one required raw request body unavailable after HTTP 429. All 60 launcher-named session artifacts and all 105 raw model responses were retrieved. B made three successful inspections and three failed submissions, exhausted six constructor requests, and never froze. Actual-target mismatch remains HOLD independently.
 files: This directory contains RESULT.md, RECONCILIATION.json, allowlists, individual request/response receipts, raw exact-version files, cross-checks and EVIDENCE-HASHES.json.
 checks: 330 unique-path download attempts; 316 HTTP 200 files; 13 HTTP 404 results and one HTTP 429 preserved; no path retried. All successful-file hashes verified. Official successor grids/levels reconcile for A's 10 and B's 2 transitions. Model response accounting and teardown records reconcile. Protected inputs unchanged.
 handoff: Review the failed source representation and the identity incident. No mechanical acceptance, gameplay efficacy claim, R10b clearance, push, retry, submission or cancellation follows from this report. Raw request 024 remains unavailable under the no-retry boundary.

## Exact provenance and collection limits

Every download used `ApiDownloadKernelOutputRequest` with `owner_slug="pranavagrawai"`, `kernel_slug="r10a-cd82-constructor-repair"`, integer `version_number=1`, and one validated `file_path`. The selector comes from the preserved provider push receipt, not from a latest-version lookup. The actual target is still different from the approved `pranavagrawai/arc3-r10a-cd82-20260916`.

The freshly collected launcher hash is `fdcfe9457f4d4642559b746278d1539860a7647c1ad8ba8ee95dd8d8d4792976`, matching the independent version-provenance review. Its 60 session paths were independently regenerated with frozen `collect_plan.py`, checked against `session-mapping.json`, and restricted to the two named CD82 sessions. Additional paths come from exact filename producers for accounting, root/child publication, GPU/model identity and teardown. No listing, bundle, arbitrary subtree, environment dump, credentials or signed redirect URL was collected.

- 330 unique paths attempted; 316 files retrieved, totaling 35,981,384 bytes.
- All 60 launcher-named session files retrieved, including IPC, both official recordings, host events, parent and child constructor results/progress/tool events.
- 104/105 raw wire requests and 105/105 raw response bodies retrieved.
- Thirteen 404s were retained for explicitly probed freeze/conditional paths. These are endpoint observations, interpreted alongside successful launcher and result records, not independent proof that arbitrary remote files do not exist.
- First phase stopped immediately on HTTP 429 at `model-runtime/solver-workspaces/raw/raw-requests/024.wire.json`. After 83 seconds, a second phase fetched only previously unattempted paths with a two-second pause between requests. It added no retry of the failed path and had no failures.
- The missing request's retained IPC message sequence, passed through the frozen logical-to-wire projection, matches the relay's message digest `d33bf96b4506da4f33905465a652b8713cdb2e027c4ae5fedd952a952f0d8cdd`. This is explicitly derived evidence, not the missing raw wire bytes. See `CALL024-DERIVED-MESSAGE-CHECK.json`.
- Failed response receipts preserve HTTP status, error type, body length/hash and timestamps. The 429 body was 82 bytes; its content was not retained. Empty 404 bodies have the empty-byte hash. Successful bodies are retained byte-for-byte. Authentication headers and redirect URLs were excluded.

`receipts/` and `receipts-remaining/` pair each exact request with a response record and raw-file hash. `ALLOWLIST.json`, `REMAINING-ALLOWLIST.json`, `COLLECTION.json` and `COLLECTION-REMAINING.json` preserve scope, order, caps and failures. The collector never approached its 340-request or 512 MiB aggregate caps.

## Observations: official progress and predictions

| Arm | Exact session | Executed IPC transitions / official successor frames | Final levels | Final state | Stop |
|---|---|---:|---:|---|---|
| A | `arm-A/cd82-fb555c5d-rmrz0g5o` | 10 / 10 | 0 | NOT_FINISHED | model_admission_budget |
| B | `arm-B/cd82-fb555c5d-b7azqbdf` | 2 / 2 | 0 | NOT_FINISHED | model_admission_budget |

Sessions are under `raw/model-runtime/solver-workspaces/`. Every transition's before/after grid matches the corresponding IPC states. Every successor grid, state and level count matches its official recording frame, in order. Both recordings end with an empty-data terminator. Launcher and `game-progress.jsonl` counts agree. No win or completed level is recorded; scorecard_run is null.

All official recording `action_input.id` fields are 0. They do not independently corroborate the IPC action labels. Action-label evidence remains IPC-bound: A executed UP, UP, DOWN, LEFT, RIGHT, SPACEBAR, SPACEBAR, LEFT, SPACEBAR, SPACEBAR; B executed RIGHT, DOWN. None was CLICK. This metadata limitation does not change the observed frame/level equality, and it prevents a stronger action-ID claim.

A has ten and B two incumbent prediction/feedback records, all `unavailable`. The collected step checks are unresolved, seven in A and two in B. No committed incumbent model score or frozen-constructor prediction accuracy is established. There are zero successful frozen queries, assimilations, unscored frozen commitments or pending frozen predictions. Action/step/digest binding, JSON-null frozen-state handling and final-pending assimilation are therefore **unexercised**, not passed or failed by this remote trace.

## Observations: constructor inspection and failed repair

B has one recorded construction attempt after its first RIGHT transition at step 1. The host event is `request_construction:REJECTED`; its result is `BUDGET_EXHAUSTED`, `capped=true`, `frozen=false`, and `worker_alive=false`. The launcher summary's `requested_construction=false` counts no completed accepted construction; it must not be read as “no attempt.” A has no constructor calls or published construction records.

The six model calls are relay calls 74–79:

1. Three `execute_python` inspections completed with return code 0. They read `/evidence/case.json`, found one visible transition, and examined its changed cells. The third inspection's complete reported output matches a fresh comparison of the exact downloaded B step-1 transition: RIGHT, 64×64 grids, 201 changed cells, including the first 80 printed tuples.
2. Three `submit_candidate` calls failed sealed ABI initialization. Each returned `candidate_source` is a JSON **string** containing a Python list expression. Static parsing yields a single top-level `Expr` and no function definitions. Sealed replay reports `NameError: name 'initialize' is not defined` for every attempt.
3. The three distinct source hashes match their recorded evaluator hashes. Raw model outputs match the tool-event calls. The next constructor requests contain the exact earlier tool feedback and matching tool-call IDs; calls 78 and 79 include the earlier NameError. Source text was preserved through transport. The subsequent submissions shortened the list expression but did not change its representation into executable function definitions.

Parent and child RESULT records agree; their tool-event files agree. Both report three inspections, three submissions, six requests, 127,911 prompt tokens and 2,844 completion tokens. The host construction duration was 84.139 seconds; maximum constructor HTTP duration was 24.700 seconds. These are below the 240-second soft wall, 245-second hard wall and 30-second POST ceiling. Six requests exhausted the request ceiling. Each request reserved at most 4,096 completion tokens; actual cumulative completion stayed below 24,576. No root or child freeze was published according to the successful result records and exact-path 404 observations.

The remote episode-level construction repair rate is **0/1**: one episode had failed submissions, and none was accepted after repair. Submission failures are **3/3**. No successful visible ABI replay, freeze or prospective frozen use occurred. The generic transport label `real-http-not-claimed-live-model` is not used alone to establish model provenance; the raw response/relay/model records provide that chain.

## Observations: requests, usage and boundaries

| Arm / role | Calls | Prompt tokens | Completion tokens |
|---|---:|---:|---:|
| A Actor | 55 | 2,720,709 | 17,111 |
| A critic | 6 | 11,903 | 408 |
| B Actor | 36 | 1,333,406 | 16,799 |
| B critic | 2 | 3,501 | 126 |
| B constructor | 6 | 127,911 | 2,844 |
| Total application | 105 | 4,197,430 | 37,288 |

All 105 response-body hashes match the launcher receipts and the parsed response ledger. Independent sums equal the launcher totals; frozen `local_accounting.accounting` gives the same 105 receipts and complete usage accounting. All 104 retrieved request message digests match their ledger declarations. The absent raw request 024 remains a collection gap despite the derived message check.

The raw server log contains 106 successful chat-completion accesses: 105 application calls plus one warmup, matching the runtime count receipts and remaining below 259. The separate `/freeze_gc` access is server memory management, not a constructor freeze. No summarizer calls occurred. Completion and request-duration caps are respected. Recorded incumbent lifecycles show 61 admitted/completed calls in A and 38 in B, with zero admissions after their deadlines and zero reasoning-present calls. Constructor calls are separately counted in B's shared purse.

Each arm received a 600-second admission allocation and a 129-call ceiling. A used 61 calls; B used 44 including construction. Both finished pending work and stopped on admission budget, within their 705-second worker watchdog. This is the approved asymmetric treatment cost, not a compute-matched efficacy comparison.

## Observations: identity and teardown

The gameplay mount receipt identifies `cd82-fb555c5d` under the competition mount, with game and metadata hashes matching the frozen registry. The earlier bootstrap LS20 preflight and the runtime result's legacy LS20 prose remain preserved; actual launcher selection, game preflight, sessions and recordings identify CD82. No LS20 gameplay is established by that bootstrap receipt.

Runtime model verification reports Qwen/Qwen3.8-27B-FP8 revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`: 80 files match the frozen expected sizes/hashes; optional `.gitattributes` is absent. The reported structure is qwen3_5 / FP8. The local model path agrees across startup arguments, raw wire requests and response identities. Official framework revision/hash receipts and shared-agent/worker hashes match the frozen contract; ARM-DIFF reports only allowed arm differences.

GPU discovery reports one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.159.04. Containment receipts report candidate and inspection checks passed. These are collected runtime observations with reconciled source/hash bindings, not a new independent physical inspection or provider-signed attestation.

Underlying shutdown scans corroborate the summaries: five owned processes before shutdown and zero after; two GPU processes before and zero after; zero remaining or unresolved listener inodes. The observer is outside the target process group/session, with consistent target identity. The first closed scan is at 5.754 seconds after TERM; seven closed scans cover 5.018 seconds and remain closed. The teardown bytes are an exact suffix of the server log. Server return code is 0; bootstrap reap reports no survivors, and the process guard reports retired cleanup. No external cancel was issued by this task; TERM is the runtime's predeclared teardown.

The earlier provider COMPLETE observation remains **current-only**, because exact string-version status lookup still failed. Numeric-version output retrieval establishes the provenance of these files, not exact-version status API success or a billing guarantee. Earlier terminal quota receipts remain in `../push-20260916/`; their SDK/CLI discrepancy is not silently resolved here.

## Interpretation and allowed next action

The returned target executed a bounded CD82 pair with coherent progress, model-response accounting and teardown evidence. Construction inspection and feedback transport were exercised. The provider repeatedly returned the wrong source representation, and the host rejected it without publishing a freeze. This resolves the first actual constructor failure boundary in this trace; it does not certify general constructor correctness or negate prior local repair tests.

The card's remote freeze/prospective-use acceptance criterion was **not met**. Its prediction of an actual B freeze did not occur. Separately, the actual-target mismatch continues to invalidate launch acceptance. Neither the 10-versus-2 action counts nor zero levels support a treatment efficacy claim. R10b remains gated.

PM/Hermes may review these existing bytes, the source-representation failure and the identity incident, then decide on separately authorized work. No retry of missing request 024, repair, repush, rename, fabricated launch.json, submission, cancellation, acceptance document or R10b launch is authorized or performed here. The unresolved collection item and 429 body-content omission are explicit; the evidence is sufficient for the descriptive failure diagnosis but is not a complete raw request archive.

## Reproduction and handoff files

From the repository root, the executed collection commands were:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B analysis/r10-prep-20260916/exact-collection-20260916/prepare_allowlist.py
/home/pranav/.local/share/uv/tools/kaggle/bin/python -u -B analysis/r10-prep-20260916/exact-collection-20260916/collect_numeric.py
/home/pranav/.local/share/uv/tools/kaggle/bin/python -u -B analysis/r10-prep-20260916/exact-collection-20260916/collect_remaining.py
python3 -B analysis/r10-prep-20260916/exact-collection-20260916/reconcile.py
```

Do not rerun collection commands: they preserve exclusive receipt directories and the no-retry boundary. `reconcile.py` is offline and can rerun against the completed collection. `RECONCILIATION.json` contains per-step comparisons, exact constructor results/source hashes, request/usage checks, identity records and teardown detail. `ACCOUNTING-TEARDOWN-CROSSCHECK.json` records the independent arithmetic/helper agreement and raw shutdown comparisons. `PROTECTED-INPUT-HASHES.json` and final verification bind the frozen sources, production controls and reviewer artifacts; none changed. All task writes are under this new collection directory.
