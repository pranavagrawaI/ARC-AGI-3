# Independent analysis: flash-next-feasibility-20260909

Richard Feynman's method: separate what the instrument recorded from the explanation attached to it. Raw artifacts were inspected and an independent provisional view formed before reading RESULT-v3.md, RESULT.md and STATUS.md. No server was launched, no tool output executed, and no existing evidence changed.

**I disagree with the recorded pass. Both v2 and v3 contain an OOM allocation failure, and their identical frozen card explicitly invalidates OOM runs.** V3 completed its eight response checks; that is a true observation, but it is not the card's licensed verdict. The appropriate card classification is **operationally invalid → retain 27B; preserve failure, no retry under this card**. A future card can distinguish recovered allocator failures from fatal OOM, prospectively. This card does not.

The most consequential findings omitted from, or misstated in, the supplied conclusions are:

- **V2 had the same OOM.** `raw/vllm-openai-server.log:488` records the same failed allocation size and free-memory value as v3 line 504. RESULT.md's “none of the Invalid-if conditions occurred” and “no OOM” are false. Its ambiguous verdict rests on an invalid operational premise.
- **The OOM recovery mechanism is unproved.** V3 logs an allocator failure and subsequently finishes autotuning. It does not identify an OOM-rejected tactic. An allocator reclaim/retry that eventually satisfies the same allocation also fits these artifacts. The recurring “Skipped 4 unsupported tactic(s)” messages appear before the OOM too; they do not establish an additional skip caused by it. Consequently the discussion of a slower chosen kernel and “almost certainly” unchanged acceptance is speculation built on an unverified event.
- **Cold inference compilation is directly recorded.** V3 server lines 620–623 name three JIT compilations during the requests. This is a demonstrated latency confound, whereas a throughput penalty from a supposedly skipped tactic is only conjectural.
- **Identity checks are weaker than complete payload verification.** Setup provenance says model verification was `identity-only`, with zero payload files checked and `payload_sha256_verified: false`; runtime layers also have `actual_sha256: null`. The model path, metadata, runtime version and PLE patch agree with the intended profile. Full weight/image byte identity was not re-established by this run.
- **The long-context correction still overclaims.** A KV capacity quotient does not demonstrate operational concurrency or 64K feasibility. “Unaffordable the moment [prompts] approach the new ceiling” does not follow even from its own 105,202-token pool arithmetic: one 65,536-token sequence is below that capacity. Conversely that arithmetic alone cannot prove it fits all other memory requirements.

## Raw reconstruction

All paths below are relative to this directory unless stated otherwise. The raw checks reconstructed content and tool arguments from every saved SSE stream, compared terminal usage with each response JSON, recomputed request SHA-256 values, and checked the actual offered schemas. The five Actor schemas require an object with a string `code`; each returned exactly that. The synthetic tool requires integer `value`, no additional properties; both versions returned `{"value": 4}`. Local `jsonschema` was unavailable, so this independent schema check inspected those complete, simple constraints directly rather than pretending to rerun the library. Actor messages and tool schemas match `actor-cases.json` exactly.

The following are observations from `raw-v3/results.jsonl`, not qualified performance estimates. Latency is client elapsed time; TTFT is first nonempty content/reasoning/tool delta. Rates were recalculated as completion tokens / elapsed seconds.

| Request | Prompt tokens | Completion tokens | TTFT (s) | Elapsed (s) | Effective tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 18 | 2 | 0.628 | 0.647 | 3.1 |
| tool | 284 | 26 | 0.819 | 0.915 | 28.4 |
| json | 33 | 11 | 0.492 | 0.549 | 20.0 |
| actor-case-0 | 10,559 | 107 | 1.742 | 2.354 | 45.4 |
| actor-case-1 | 11,228 | 72 | 0.874 | 1.278 | 56.3 |
| actor-case-5 | 17,578 | 78 | 1.375 | 1.835 | 42.5 |
| actor-case-12 | 21,130 | 282 | 1.598 | 3.551 | 79.4 |
| actor-case-13 | 21,257 | 79 | 1.606 | 2.032 | 38.9 |

Totals: **82,087 prompt tokens, 657 completion tokens, 13.162 s of summed client request latency**. All eight SSE streams terminate with usage and `[DONE]`. All reasoning fields are empty and usage reports zero reasoning tokens. All five Actor calls finish as `tool_calls` naming offered `execute_python`. This establishes response envelope/schema observations; no code execution, action validity, or solve quality was checked. “Schema-valid code” means a string here, not a functioning program.

V2 has two passing rows in `results.jsonl`, plus a complete failed third response saved separately. Its totals are **327 prompt tokens and 39 completion tokens** across three requests. The JSON request takes 0.515 s, TTFT 0.457 s, returns `{"key": 4}`, then fails the exact-match assertion. The failure is accounted for despite its absence from results.jsonl. No Actor requests were issued.

### Native counters and unused timing evidence

The named MTP counters start at zero in both versions. Recomputed v3 before/after deltas are **222 drafts, 665 draft tokens, 441 accepted tokens**. Thus 441 / 665 = **66.3%** rounded to one decimal. That arithmetic is correct. It is a pooled, token-weighted observation over eight heterogeneous requests, including three synthetic requests, not an Actor-only rate or a stable profile parameter. There are no per-request metric snapshots or repetitions with which to estimate between-request variability. Calling it a “meaningful sample” does not supply that missing evidence. Draft decisions are dependent, so 665 is not 665 independent experiments.

Previously undiscussed position counters are **176, 146, 119 accepted tokens** at positions 0, 1, 2; they sum to 441. Dividing by 222 drafts gives descriptive per-draft yields of 79.3%, 65.8%, 53.6%, respectively, not conditional acceptance probabilities at each position: 665 is one short of 222 × 3, and per-position proposal denominators are not supplied. Accepted yield is 441 / 222 = 1.99 tokens per draft. Neither quantity is a measured speedup.

V2 deltas are 11 drafts, 32 proposed and 32 accepted tokens. Describing its completions as “deterministic” is unwarranted: requests specify temperature 1.0, top-p 0.95 and top-k 20. Its observed 32/32 is reportable with the tiny-sample qualification; there is no statistical principle that prohibits writing the fraction while allowing the v3 fraction.

Other v3 metric deltas cross-check the client and identify costs:

| Counter or histogram sum | Delta | What it adds |
| --- | ---: | --- |
| prompt tokens / local compute | 82,087 | Matches client exactly; cached and transferred token counts are zero |
| generation tokens | 657 | Matches client exactly |
| successful requests, stop | 8 | Length, abort, error and repetition counters remain zero |
| server end-to-end request time | 13.149 s | Close to 13.162 s client sum |
| queue time | 0.022 s | Queuing is negligible in this serial test |
| prefill time | 8.267 s | About 63% of server end-to-end time |
| decode time | 4.330 s | About 33% of server end-to-end time |
| server TTFT | 8.819 s | Different boundary from first visible client tool/content delta |
| inter-token latency count | 222 | Does not represent 649 separate visible token intervals under speculation |

These are server-defined timing observations, including cold work; they are not a clean kernel decomposition. JIT warnings name `_qsa_sparse_paged_gqa_splitk_kernel`, `_qsa_mqa_paged_kernel`, and `_expand_qsa_indices_kernel` at 12:35:15–17. No timestamped per-delta client trace or repeated warm trials separates compilation from processing for each fixture. The periodic log's 74.9% acceptance uses only 152/203 tokens, a partial interval, and does not contradict the final 441/665. Metrics also show zero preemptions; idle KV gauges before/after cannot establish peak KV use.

### Identity, setup, resources and cleanup

`vllm-server-identity.json` records the intended Kaggle model version-1 path, BF16 dtype, `modelopt_fp4`, TP=1, 32,768 context, 5 GiB KV reservation, 8 maximum sequences, 8,192 batch tokens, chunked prefill, prefix caching disabled and MTP depth 3. The server log independently reports vLLM `0.1.dev20073+g8e685d198`; the longer fingerprint in the results is consistent with that version. Provenance records the intended HF revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` and verified PLE target patch hash. This is consistent identity evidence, not evidence of wrong identity. Its explicit fast-path payload verification omissions must remain visible.

`vllm-server-preflight.json` is **skipped**, and its explanation refers to an “authorized one-game smoke” that this client does not perform. Its advertised final teardown file `vllm-metrics-final.prom` is absent; our `metrics-after.prom` provides the actual end counters. Provenance also defers final process/GPU evidence to teardown, but the retained cleanup record is process/port evidence, not a post-cleanup GPU sample. These stale donor descriptions must not be mistaken for additional completed checks.

Ready minus server-start epochs is **410.057 s**, agreeing with the 410.056 s monotonic measurement. However total setup is **514.107 s**, including **103.938 s runtime extraction**. V2 likewise has 450.067 s server readiness but **551.686 s total setup**. Both satisfy 1800 s; RESULT.md's “Setup cost 450 s” understates setup, and cross-version startup variation is not a controlled speed comparison.

V3 has **106 resource samples spanning 526.534 s**, all identifying NVIDIA RTX PRO 6000 Blackwell Server Edition, total 97,887 MiB. Sampled GPU utilization spans 0–100%; whole-run statistics would mix extraction, loading, autotuning and requests. Maximum sampled memory is 96,628 MiB, leaving 1,259 MiB, but the OOM reports only **40.375 MiB free** while requesting **488 MiB**. Those are distinct instruments and moments, not alternative estimates of one stable margin. Only three resource samples follow readiness; they report 92,700, 94,860 and 94,860 MiB used. Thus “about 1.2 GiB during steady-state serving” is unsupported: 1.23 GiB is the all-run sampled minimum free space, while those three samples leave about 5.1, 3.0 and 3.0 GiB. None establishes an instantaneous serving minimum. Host MemAvailable reaches a sampled minimum of about **118.34 GiB**; it is not per-process resident memory.

Server lines 439, 479–480 and 560 report model loading using 81.8 GiB, explicit KV reservation with memory profiling skipped, capacity 105,202 tokens / nominal 3.21× concurrency at 32,768, and graph capture using 0.36 GiB. “Model loading memory” should retain that label rather than become an exact standalone weights budget.

Cleanup in both versions records matched owned identity, SIGTERM, a separately signaled descendant, no survivors and a closed port. That meets the recorded cleanup gate. But v3 lines 633–648 also show forced EngineCore termination and resource-tracker warnings about two semaphore and two shared-memory objects. V2 has the analogous warnings. This is successful process retirement, not proof of wholly graceful shutdown or verified removal of every IPC resource. The warnings alone do not establish a failed cleanup under this card.

## Claim audit and competing explanations

**The v3 result table is numerically correct.** Its decimal places are reasonable for reporting particular observations; the generality attached to them is the problem. Five selected fixtures, one completion apiece, do not characterize Actor latency distributions, sustained throughput, or broader reliability. Reasoning suppression was observed on these eight inputs; it was not established universally. No matched MTP-off run identifies MTP's latency contribution. No artifact here supports assertions about 106 calls from another session.

**The v2 prompt diagnosis is plausible, not causal proof.** The original wording admits a key/value reading, and `json_object` requested JSON rather than a key-specific schema. Therefore this is not evidence that JSON syntax enforcement broke. But ordinary instruction-following error also fits `{"key": 4}`. One success after rewording does not establish that wording alone caused the change, or that the original would invariably fail. RESULT-v3's “the gate was never the defect; the v2 prompt was” and RESULT.md's “Nothing here would change if ... worded correctly” exceed the experiment. The direct fact is exact-output failure, followed by success on a changed prompt in a new run; both sessions also had an operational invalidator.

**Context accounting is correct; the deployment inference is not settled.** The largest prompt leaves 32,768 − 21,257 = **11,511 tokens**, or **10,487** after reserving the full 1,024 output cap. The maximum actually returned total is 21,412 tokens (case 12). The earlier rough 12k estimate in RESULT.md is materially low. “First direct measurement” cannot be established from these artifacts alone; retained source usage exists and the card correctly distinguishes its tokenizer provenance.

`ANALYZER_CONTEXT = 32_768` is indeed hardcoded in donor line 129 and used by the server command and exported client environment. The listed coupled sites exist; disabled-chunking guards are inactive here. Changing the profile requires an implementation/configuration variant, but “means patching Keith's donor constant” is a workflow choice, not a unique technical route. The model log even records the draft model's length being overridden from 262,144 to 32,768; this establishes configuration distinction, not tested higher-context service.

The pool arithmetic 105,202 / 65,536 ≈ 1.61 and / 21,257 ≈ 4.95 is only nominal capacity arithmetic. It excludes completion reservation and does not demonstrate fractional or integer concurrent requests actually served. This client is serial despite `max_num_seqs=8`. Preserving a capacity ratio by doubling the pool is not the same as proving that a +5 GiB allocation must fail after all other allocations settle. A startup free-memory trough and a fixed memory-pool budget cannot certify either affordability or impossibility of an unrun configuration. Likewise assertions that scratch demand is independent of context need the relevant kernel/allocation implementation or a controlled change; this log does not establish that dependency.

**V1 cannot be independently verified from the supplied raw retention.** STATUS.md names `raw/flash-next-feasibility-20260909.log` as its papermill evidence, but that file now contains v2 execution, not the claimed missing-kernelspec failure. Neither raw directory retains a v1 notebook/failure trace. Current build metadata cannot prove the contents of the failed original. The kernelspec explanation remains reported history; “no GPU quota” also lacks a retained allocation/quota record. V3 log ends in notebook conversion around 541 s, consistent with successful completion, but the exact platform `COMPLETE` timestamp and 546 s status observation are not retained as an independent status payload here.

The card, donor files and Actor fixtures are byte-identical between **v2 and v3**; recomputed hashes agree with the write-ups. The probe diff contains exactly the three described changes. Equality with v1, scratchpad gate-test results and launch authorization cannot be independently authenticated from these raw artifacts. STATUS records explicit v3 authorization, so I do not infer an unauthorized automatic relaunch merely from there being another version.

## Frozen-card compliance

| Clause | Finding |
| --- | --- |
| Prediction | V3 meets the observable eight-response prediction within the setup budget. This does not override Invalid-if. |
| Stop | V2 stops on its failed exact-output check; v3 stops after eight responses. No server restart is evident. Recovered OOM is in Invalid-if, not an explicit immediate Stop clause. |
| Wrong identity | No mismatch found. Live identity is consistent, with fast-path verification limits above. |
| Missing accounting / terminal usage | No gap found for issued requests; v2's failed response is outside results.jsonl but fully retained. |
| Timeout | None observed; actual requests and setup fit their bounds. |
| OOM | **Triggered in both v2 and v3. No recovered-OOM exception exists.** |
| Failed cleanup | Not established: owned-process/port checks pass, with shutdown qualifications above. |
| Cap | V3 issues eight bounded requests; v2 issues three. Client contains no tool execution/gameplay/submission or automatic restart. Two GPU-serving sessions exist under the same card identity; they cannot be represented as one session's quota exposure. STATUS reports fresh user authorization for v3, but an unchanged single-session card is not a fresh versioned precommitment. |
| Result map | **Operational failure / invalid in both versions.** The v3 Pass branch and v2 valid-Ambiguous branch are not licensed by the OOM wording. |

The client writes `verdict.json` before final metrics and cleanup, checks no OOM log condition, and never revises the pass afterward. Consequently its pass is a **response-gate result**, not a complete card assessor. The whitespace change also leaves a non-whitespace reasoning abort absent from the card; it narrows, rather than eliminates, that mismatch. Allowing Actor prose to continue is consistent with the Stop clause, but a future prose outcome would still need interpretation against the prediction's “including offered Actor tools”; unconditional file-level pass cannot make that judgment. Neither dormant gate issue caused this run's observed result.

The observations should be retained intact. They show successful response checks after a recovered startup allocation failure, and provide useful diagnostic counters. They do not license capability or speed conclusions under the experiment's own explicit invalidity rule. The decisive correction is to the verdict and its evidentiary status, not to erase the eight responses or invent a model failure.

Audit scope: inventoried both raw directories; reconstructed every saved response/SSE pair and request hash; compared fixtures and donor/card hashes; read client gates, identity, preflight, provenance, resource series, metric deltas and relevant startup/inference/shutdown log sections; then checked all three supplied conclusions. “Unexamined” above means absent from those conclusions, not a claim to know what every other investigator privately inspected. Shared-board consultation was attempted but failed with `unable to open database file`.
