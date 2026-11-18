# What we learned: inference performance

> Coordinator closing update: a concurrent [independent checker review](../../../analysis/deployment-probes/20260909-experiment-checks/REVIEW.md) now reports 49 tests and corrected `reviewed-final/` audits. Its historical SSE reconciliation covers all eleven issued v2/v3 responses and retains the OOM invalidation. Full payload hashes, cross-model tokenizer parity, and comparative gameplay remain unverified. Earlier audit links below intentionally preserve their generation; reviewed report hashes/text are archived in [source updates](../sources/reviewed-source-updates.json).

Evidence reviewed 2026-09-09. Scope: the retained Qwen/SGLang FP8 serving experiments, Actor replay and timing, concurrency, prefix caching, and Flash-Next feasibility. This appendix records decisions already supported by evidence; it proposes no new solver architecture.

**Retained decision: keep the measured 27B FP8 baseline and chunk size 2048; MTP remains parked, and Flash-Next has no valid PASS.** This is a disposition of these experiments, not a claim that every current solver package has identical settings. The durable gains are a working offline sampling repair, exact-request capture, measurements that distinguish request cost from useful action cost, and a corrected cache/concurrency account. No experiment here establishes a serving change that improves ARC solve quality.

## Evidence order and coverage

Use raw requests, counters and ledgers to test claims; use later reassessments ahead of original result prose. In particular, [prefix REASSESSMENT](../../../analysis/deployment-probes/20260907-prefix-cache/REASSESSMENT.md) supersedes graph/repeatability/cache explanations in its original RESULT-run; [Flash independent analysis](../../../analysis/deployment-probes/20260909-flash-next-feasibility/INDEPENDENT-ANALYSIS-astra.md) and [operational audit](../../../analysis/deployment-probes/20260909-experiment-checks/flash-v3.json) supersede the original PASS; [runner overlap](../../../analysis/competitor-transfer-20260906/runner-overlap/REPORT.md) supersedes the inference of global serialization from the earlier fake-worker test.

Coverage is strongest for September 5–9. Earlier FP8 preparation is visible through frozen cards and the retained August 31 remote source, but this checkout does not contain all earlier raw runs named in those cards. This is not a complete history of every BF16/FP8 backend attempt, provider comparison, or all solver research. No current online source or model was queried. Later live prompt counts are provider-reported, not retokenized for Flash-Next. Full summary-boundary request bodies remain absent from the replay corpus. The 95k inputs are mechanical stress, not natural long Actor histories.

## The units that must remain separate

| Quantity | What the evidence measures | What it cannot stand for |
|---|---|---|
| Steady decode tokens/s | In the native fixed controls, `(completion_tokens − first_batch_tokens) / tokenizer_decode_span_s`, from server tokenizer first output batch to finished | Pure GPU kernel time, full HTTP rate, or game speed |
| Effective tokens/s | Completion tokens divided by elapsed request time, or summed completion tokens divided by summed call times | Equal-work acceleration when natural output lengths differ |
| Request latency / TTFT | Client dispatch to terminal response; Flash TTFT is first nonempty content, reasoning or tool delta | Workload-ready latency; a raw HTTP chunk need not be a generated token |
| Finite-set throughput | Total tokens / common set elapsed time; observed at HTTP C1/C2/C4 | Individual deadlines, independent-game throughput, or host admission qualification |
| Ready-to-completion | Common workload-ready instant to each completion, including executor waiting before HTTP dispatch | Actual game deadline compliance; the 180 s threshold was diagnostic |
| Actor decision | Multiple model → CodeAct → model rounds until a batch is submitted | One model request; host decisions serving queued actions can be cheap |
| Environment action throughput | Executed actions / gameplay wall time | Level progress: changing a frame is not solving a level |
| Setup / allocation time | Bootstrap, readiness, queue, finalization and containment windows | Inference time or an independently metered GPU bill |

The definitions are explicit in [native result metadata](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/raw/mtp-sampling-repair-20260906/baseline/fixed-1-result.json), [Gregg request accounting](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/raw/baseline-cost-gregg-20260906/batch-long95000-4-0.json), and [Flash raw reconstruction](../../../analysis/deployment-probes/20260909-flash-next-feasibility/INDEPENDENT-ANALYSIS-astra.md).

## Chronology and experiment ledger

Each row separates actual exposure from implementation or intended exposure. “Retained” can mean a diagnostic fact or working instrument, not deployment selection.

| Candidate / question | Implementation and actual exposure | Outcome and validity | Retained gain; unresolved question |
|---|---|---|---|
| August 31 R8-L3B: preserve FP8 request behavior while retiring the server cleanly? | Pinned SGLang wheel shutdown ownership/order patch. The card reports rejected generations for metadata filename, provider-normalized wheel filename, and mount-root assumptions; generation 4 uses bounded exact-name/hash discovery. | Predecessor v12 reportedly passed the request path and retired root-0/TERM-only, but failed strict teardown taxonomy. Earlier raw evidence is not all present here; the card is historical evidence, not a fresh qualification. | Runtime/lifecycle distinction and byte identity became inherited controls. No speed effect established. [Card](../../../deployment/kaggle/PROTOTYPE-r8l3b-qwen38-fp8-patched-lifecycle-EXPERIMENT-CARD.md) |
| Sep 5 isolated engine v1/v2: does FP8 tuning or shallow MTP improve decode? | V1 requested RTX PRO 6000 but actually received P100; hardware gate stopped before loading. V2 received SM120 and matched model, then failed serializing Transformers `BatchEncoding` before any server arm. | Neither produced engine measurements. Allocation mismatch and harness API assumption are separate failures. | Check observed hardware and real tokenizer serialization, not notebook metadata or a mocked preparation path. [v1](../../../analysis/deployment-probes/20260905-isolated-engine-v1/RESULT.md), [v2](../../../analysis/deployment-probes/20260905-isolated-engine-v2/RESULT.md) |
| Sep 5 isolated engine v3 | Three measured arms on RTX PRO 6000: baseline auto backend, five shape-gated explicit Triton FP8 configs, shallow EAGLE 1-step/top-k-1/two draft-token setting; identical 81-token input, greedy sampling, fixed 1024-token controls. | MTP +58.6% decode; tuned Triton +3.5%, below the >10% bar. Clean containment. Output tokens diverged across arms although all natural outputs reached the same arithmetic answer. | MTP earned a matched Actor comparison, not promotion. Kernel-numerics explanation for divergence is plausible, not isolated proof. Baseline auto vs explicit tuned Triton is not tuned-vs-untuned Triton. [Result](../../../analysis/deployment-probes/20260905-isolated-engine-v3/RESULT-run.md) |
| Sep 5 q6 speed / quick-host | Speed: 10 actions, 6 Actor decisions, 106 recorded calls. Quick-host: repaired packaged host, 10 actions, 8 decisions, 53 calls. | Speed run failed post-play schema accounting and missed one summarizer call. Quick-host passed 53+1 warmup reconciliation; no summary occurred. Timing observations differ, but this is not a randomized host-change speed trial. | Multi-call decision cost is real; fixes qualified short host execution. Cause of between-run behavioral speed difference and long-run quality were not isolated. [Speed](../../../analysis/deployment-probes/20260905-q6-speed-probe/RESULT.md), [quick-host](../../../analysis/deployment-probes/20260905-q6-quick-host/RESULT-run.md) |
| Sep 6 Actor capture | Transparent recorder saved 14 exact request/response pairs, 10,581–21,279 prompt tokens; all HTTP 200, 14 recorded calls + one warmup reconciled. | Zero environment actions; three responses named unoffered tools. Instrumentation succeeded, gameplay failed to produce a decision within the worker window. | Valid failure-labelled prompt corpus; no summary or forced-action coverage. It is not a corpus of successful Actor behavior. [Result](../../../analysis/deployment-probes/20260906-actor-capture/RESULT-run.md) |
| Sep 6 Actor MTP replay | Baseline completed all 14 natural replays; MTP server started then crashed on its first generation with real `temperature=1, top_k=20, top_p=.95`. | FlashInfer top-k renormalization tried offline JIT and failed with misleading “sm75 or higher” error on SM120. No paired speed estimate. | Sampling-specific executable path must be exercised. A greedy synthetic warmup did not qualify it. [Result](../../../analysis/deployment-probes/20260906-actor-mtp/RESULT-run.md) |
| Sep 6 q6 MTP integrated pilot | Runtime warmup succeeded; first real solver request killed scheduler on the same sampling path. | 0 committed actions, 0 recorded solver model calls, no established speculative work; elapsed 1194.3 s largely included a 900 s worker wait after engine failure. | The integrated configuration was unusable. Neither prediction-feedback quality nor MTP speed was tested. Warmup sampling difference is inference because its body was not retained. [Result](../../../analysis/deployment-probes/20260906-q6-mtp-integration/RESULT-run.md) |
| Sep 6 MTP sampling repair | Routed renormalization to precompiled `sgl_kernel`, disabling FlashInfer JIT; numerical gate on SM120, stochastic execution gate, 28 natural replays plus fixed native controls. | Repair worked. Long-Actor fixed decode regressed about 29.7%; natural pooled effective rate fell 28.6%. Both arms retired cleanly; zero natural tool-schema errors in this pass. | Keep repair as qualified capability; retain baseline, park MTP. Cause of fixed-control regression and hybrid rejected-state correctness remain unresolved. [Corrected result](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/RESULT-run.md) |
| Sep 6 baseline-cost | Context sweep, cold/continuation pair, same four captured prompts at C1/C2/C4, two orders, target traces/metrics. | 95k prefill 70.8% of request time; real prefix reuse; captured C4 set about 64% faster than C1 with server overlap. Clean lifecycle, `promotion:false`. | Cost decomposition and cache reuse are measured. Isolated same-game HTTP concurrency does not establish independent-game or host performance. [Result](../../../analysis/deployment-probes/20260906-baseline-cost/RESULT-run.md) |
| Sep 6 Gregg revision | Independently repeated captured set, added mechanical 95k finite set and common-ready completion accounting. | Captured batching replicated within about 1%; 95k C4 improved makespan about 22% but worsened median ready-to-completion. All requests passed 180 s diagnostic threshold. | Throughput and deadline verdict can disagree; no backend or host change selected. No hardware-counter saturation diagnosis. [Result](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/RESULT-run.md) |
| Sep 6 host concurrency correction | Earlier two-instance fake-worker check serialized decide intervals; later actual packaged worker/headless + local NOOA loopback probe allowed A's successor request to overlap B's first request. | Peak HTTP in-flight 2 without A calling decide twice. CPU/scripted-response evidence, no GPU or real game. | `decide()` semaphore is not global HTTP admission. Real SGLang scheduling/deadlines still require their own evidence. [Earlier check](../../../analysis/deployment-probes/20260906-q6-concurrency-feasibility/RESULT.md), [counterexample](../../../analysis/competitor-transfer-20260906/runner-overlap/REPORT.md) |
| Sep 6 sustained summary-boundary run | 100 ls20 actions, 153 Actor + 2 summarizer calls, two exact-input summary verifications and resumed play. | 0 levels; 155 calls + warmup reconciled. Outer exit 74 was stale post-run chassis validation, not interpreter finalization. Saved-data repair accepts it; original deployment status remains ERROR. | Strong latency/behavior evidence and live summary liveness; not summary-quality or deployment promotion. [Result and correction](../../../analysis/deployment-probes/20260906-q6-summary-boundary/RESULT-run.md), [exit review](../../../analysis/deployment-probes/20260906-q6-summary-boundary/exit-review/RESULT.md) |
| Sep 7 prefix-cache ABBA and reassessment | Four complete lifecycles 2048/4096/4096/2048; 148 probe-owned admissions, fixed 256 outputs, exact successors, natural checks every arm, C1/C4, 95k stress. | Original ERROR on one unclassified INFO shutdown line; original assessor also globbed 36 files where it expected four. Separate read-only reassessment accepts that exact line with containment gates and finds no ≥10% candidate gain. | Retain 2048. Prefill graph settings changed; checkpoint availability explains observed cache counts. Raw evidence and original failure remain intact. [Reassessment](../../../analysis/deployment-probes/20260907-prefix-cache/REASSESSMENT.md) |
| Sep 9 Flash-Next v1/v2/v3 | Donor vLLM NVFP4 profile, 32,768 ceiling, prefix cache off, native MTP depth 3. V1 kernelspec failure is reported history without retained v1 raw trace. V2 issued 3/8 cases; v3 issued 8/8, including five Actor fixtures. | V2 exact JSON-key check failed; v3 response checks passed. Both logs contain OOM, an unconditional card invalidator. Original v3 PASS withdrawn. | Preserve response/counter observations; retain 27B. No matched 27B/MTP-off comparison, long-context service, executed tools, gameplay or promotion. [Independent audit](../../../analysis/deployment-probes/20260909-flash-next-feasibility/INDEPENDENT-ANALYSIS-astra.md), [current result](../../../analysis/deployment-probes/20260909-flash-next-feasibility/RESULT-v3.md) |

## Quantitative evidence that determines the decisions

### MTP: workload transfer failed; acceptance is not speed

Native fixed controls generate 1024 tokens. Rates below use the server tokenizer interval, not request elapsed time.

| Workload | Baseline repeats, tokens/s | Treatment repeats, tokens/s | Decision |
|---|---:|---:|---|
| 81-token math input, tuned Triton | 45.134 / 45.133 | 46.725 / 46.725 | +3.5%, reject against >10% bar |
| Same input, shallow MTP | 45.134 / 45.133 | 71.574 / 71.568 | +58.6%, synthetic diagnostic win |
| 21,279-token Actor input, repaired shallow MTP | 38.288 / 38.287 | 26.921 / 26.910 | −29.7%, retain baseline |

Sources: [engine result](../../../analysis/deployment-probes/20260905-isolated-engine-v3/RESULT-run.md), [raw engine baseline](../../../analysis/deployment-probes/20260905-isolated-engine-v3/raw/isolated-engine-20260905/baseline/fixed-1-result.json), [raw engine MTP](../../../analysis/deployment-probes/20260905-isolated-engine-v3/raw/isolated-engine-20260905/shallow-mtp/fixed-1-result.json), [raw Actor baseline](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/raw/mtp-sampling-repair-20260906/baseline/fixed-1-result.json), [raw Actor MTP](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/raw/mtp-sampling-repair-20260906/shallow-mtp/fixed-1-result.json).

**Critical correction:** both fixed comparisons are greedy. The [retained repaired request](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/raw/mtp-sampling-repair-20260906/shallow-mtp/fixed-1-request.json) says temperature 0, top-k −1, top-p 1. Stochastic renormalization therefore cannot explain this fixed-control slowdown. Context, content, runtime differences and fixed arm order were not independently varied; the cause is unresolved. Nearly identical repeats do not exclude a systematic confound.

**Additional raw-ledger correction:** engine-v3 prose interprets `spec_accept_length=1.950` as approximately 97.5% draft acceptance. Its first fixed response instead records **498 correct / 525 proposed drafts = 94.857%**, matching `spec_accept_rate=0.948571`. Dividing accepted length by the 2.0 ceiling is a different statistic. The repaired long control records **488/536 = 91.045%**, `spec_accept_length=1.910448`, yet runs slower. Neither acceptance statistic measures net saved time. The original assertion that a 1.59× speedup versus 1.95× accepted length quantifies *the* draft-forward overhead is too specific; it does not isolate verification/state/scheduling overhead.

Natural repaired replays produced 3573 baseline tokens in 121.0 s (29.53 effective tokens/s), versus 3174 in 150.6 s (21.07). Case 0005 alone produced 159 versus 656 tokens. Natural paired latency is an experience outcome under stochastic generation, not an equal-output-work benchmark. Target numerical errors were 7.45e−09 for top-k and 1.49e−08 for top-p renormalization. These qualify the narrow repair, not full speculative state correctness. [Corrected repair result](../../../analysis/deployment-probes/20260906-mtp-sampling-repair/RESULT-run.md).

Capture/replay invalid-tool counts of 3, then 2, then 0 baseline outputs over the same 14 prompts, at differing positions, undermine a deterministic case-specific explanation. They do not rule out prompt-dependent failure probabilities or establish a reliability rate. “All natural outputs schema-valid” also does not mean their Python was executed or useful.

### Context, cache and concurrency: three different costs

The baseline sweep used 256 outputs and three warmed repeats in alternating length order:

| Input tokens | Median decode tokens/s | Request elapsed s | Forward-to-prefill-finished s | Prefill share |
|---|---:|---:|---:|---:|
| 1024 | 44.52 | 5.83 | 0.101 | 1.7% |
| 10,240 | 41.39 | 7.28 | 1.112 | 15.3% |
| 21,279 | 38.11 | 9.42 | 2.725 | 28.9% |
| 95,000 mechanical | 25.34 | 34.50 | 24.426 | 70.8% |

This establishes prefill dominance for the 95k/256-output workload; it does not make decode degradation negligible (44.52→25.34 is substantial) or establish hardware bandwidth saturation. Queue time in this isolated sweep was about 0.2–0.4 ms. Kernel traces locate execution time; without counters they do not identify arithmetic versus bandwidth saturation. [Baseline result](../../../analysis/deployment-probes/20260906-baseline-cost/RESULT-run.md).

The initial natural continuation reused **18,432/21,279 = 86.62%** of the successor prompt. The token LCP was 19,708, not the cache count. Cold/warm latencies of 5.35/2.41 s generated 100/57 tokens, so “2.2× faster from caching” is not a controlled inference. Later fixed-output prefix controls supplied the appropriate comparison.

Gregg repetition 0, recomputed from per-request common-ready fields:

| Workload | C | Set elapsed s | Median ready→completion s | Max ready→completion s |
|---|---:|---:|---:|---:|
| Captured | 1 | 30.142 | 19.08 | 30.14 |
| Captured | 2 | 18.602 | 14.44 | 18.60 |
| Captured | 4 | 10.765 | 10.76 | 10.76 |
| Mechanical 95k | 1 | 133.652 | 84.84 | 133.65 |
| Mechanical 95k | 2 | 113.608 | 86.07 | 113.61 |
| Mechanical 95k | 4 | 103.832 | 103.83 | 103.83 |

Raw finite-set records: [captured C1](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/raw/baseline-cost-gregg-20260906/batch-captured-1-0.json), [captured C4](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/raw/baseline-cost-gregg-20260906/batch-captured-4-0.json), [95k C1](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/raw/baseline-cost-gregg-20260906/batch-long95000-1-0.json), [95k C4](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/raw/baseline-cost-gregg-20260906/batch-long95000-4-0.json); [result with repetitions and server-metric analysis](../../../analysis/deployment-probes/20260906-baseline-cost-gregg/RESULT-run.md).

The first baseline run reported captured aggregate throughput about 34 tokens/s at C1 versus 94 at C4. Gregg independently reproduced the set times. At 95k, C4 reduces total set time **22.31%** but worsens median ready-to-completion **22.38%** using the rounded medians. It improves the last completion while delaying the typical completion. All finished within 180 s, so no actual deadline failure was demonstrated. The server reached 1/2/4 running requests in corresponding windows; that is distinct from merely setting a client concurrency limit. The same-game prompts share prefixes, and the 95k set is repeated content: neither is an independent-game workload sample.

### Prefix ABBA: retain 2048 and the corrected checkpoint explanation

| Finite set | 2048 arm times, s | 4096 arm times, s |
|---|---:|---:|
| Captured C1 | 30.092 / 30.188 | 31.129 / 31.142 |
| Captured C4 | 11.599 / 10.695 | 11.003 / 10.999 |
| Mechanical 95k C1 | 134.108 / 133.993 | 130.050 / 130.075 |
| Mechanical 95k C4 | 103.560 / 103.523 | 98.873 / 98.868 |

The roughly 3–4.5% mechanical gains miss the 10% bar; captured C1 regresses. Captured C4 baseline spread is about 8%, contradicting the original blanket “~0.1% repeatability.” Natural validation records unknown-tool counts 1/0/1/1 across the arms; this tiny sample does not estimate a treatment quality effect. [Reassessment and exact evidence inventory](../../../analysis/deployment-probes/20260907-prefix-cache/reassessment-v1/REASSESSMENT.json).

For the long captured pair, fixed 256-output cold latencies were about 9.44–9.46 s. Predecessor-warmed latencies were 7.19/7.06 s at 2048 and 7.49/7.49 s at 4096. Raw responses confirm **18,432 cached tokens in both 2048 arms, 16,384 in both 4096 arms**. For the short pair both reused 8192. [Arm 0 long successor](../../../analysis/deployment-probes/20260907-prefix-cache/raw/prefix-cache-20260907/arm-0-chunk-2048/normal_long_12_13-predecessor-successor-result.json), [arm 1](../../../analysis/deployment-probes/20260907-prefix-cache/raw/prefix-cache-20260907/arm-1-chunk-4096/normal_long_12_13-predecessor-successor-result.json).

The corrected mechanism is **availability of matching recurrent-state checkpoints**, not a universal rule to round every common prefix down to the configured chunk multiple. With LCP 19,708, the latest available checkpoints in these cold-predecessor paths were 18,432 and 16,384. A CPU execution of the pinned checkpoint-position method matched all 12 retained counts and produced a counterexample: known branch 19,648 can be selected despite not being a multiple of 4096. That is arithmetic/source reproduction, not a GPU state-copy or eviction test. [Checkpoint evidence](../../../analysis/deployment-probes/20260907-prefix-cache/reassessment-v1/CACHE-CHECKPOINT.json).

The raw resolved policies also change prefill CUDA graph `max_bs` from 2048 to 4096 and add larger captured shapes. Thus this is a **chunk-configuration comparison**, not an isolated prefill kernel treatment. The original “all graph settings identical” statement is false. The separately reviewed classifier accepted one exact INFO shutdown line while preserving successful containment, quiet window, TERM-only and return-code gates. It did not rewrite the original ERROR or excuse arbitrary teardown failures. The reassessment reports all 602 original files unchanged and 25 local checks; those checks qualify the reassessment, not another GPU run. [Reassessment](../../../analysis/deployment-probes/20260907-prefix-cache/REASSESSMENT.md).

### Actor time: faster tokens do not remove repeated model crossings

The sustained ls20 [raw call ledger](../../../analysis/deployment-probes/20260906-q6-summary-boundary/raw/nooa-join-public3/solver-workspaces/ls20-8md_11v1/ipc/model_calls.jsonl) independently recomputes to:

- **153 Actor calls, 84,720 completion tokens, 2979.779 summed Actor seconds = 28.432 effective tokens/s.** Median call 14.409 s, nearest-rank p90 41.970 s, max 88.255 s.
- **11 calls below 20k prompt tokens: 35.010 pooled effective tokens/s; 27 calls at or above 80k: 20.266.** These are observational prompt bands, not a controlled context intervention.
- **Turn 30: 23 Actor calls and 506.989 summed Actor seconds** before the reported single RIGHT commitment.
- **Two summarizer calls: 111.140 summed seconds; peak prompt 107,615 tokens.** All 155 calls reported zero reasoning tokens.

The [gameplay report](../../../analysis/solver-diagnosis-20260906/REPORT.md) records 100 actions from 40 batches in 3005.16 gameplay seconds, 99 grid changes, zero completed levels. Derived action rate is **0.03328 actions/s ≈ 2.00 actions/min**, 30.05 gameplay seconds per executed action, 2.5 executed actions per submitted batch and 0.654 actions per Actor call. These averages are not latency distributions. Summed Actor plus summary call time exceeds gameplay wall; these durations have different boundaries and must not be stacked as disjoint wall-time shares.

Earlier speed-probe medians were **8.3 s per recorded model call versus 140.6 s per Actor decision**; one decision took 56 calls/720.2 s. The short repaired host subsequently used 53 calls instead of 106 for 10 executed actions, but changed host behavior and stochastic trajectories prevent attributing the entire improvement to one repair. Its table labels 986 s gameplay although its header also calls 986 s probe elapsed: use the 669 s model total as a model measure and do not silently use that table to infer a precise host-overhead decomposition. [Speed result](../../../analysis/deployment-probes/20260905-q6-speed-probe/RESULT.md), [quick-host result](../../../analysis/deployment-probes/20260905-q6-quick-host/RESULT-run.md).

The actionable historical finding is the multiplication of requests before a commitment, not a new proposed control architecture. The 32k environment variable did not constrain these full requests: the observed 107,615-token prompt falsifies that reading. Summary liveness and exact serialization were shown twice; preservation of useful distinctions was not measured.

### Flash-Next: retain observations, withdraw feasibility and speed claims

The donor profile is Qwen3.8-Flash-Next-NVFP4 on vLLM `0.1.dev20073+g8e685d198-bf09729e`, configured ceiling 32,768, prefix caching disabled, native MTP depth 3. This is distinct from the 27B SGLang FP8 path. The [independent audit](../../../analysis/deployment-probes/20260909-flash-next-feasibility/INDEPENDENT-ANALYSIS-astra.md) establishes that both v2 and v3 violate the frozen card's OOM condition. V3's allocator log requests **511,705,088 bytes with 42,336,256 free**. Eight completed responses do not override that invalidator. [Raw server log](../../../analysis/deployment-probes/20260909-flash-next-feasibility/raw-v3/vllm-openai-server.log), [machine-readable audit](../../../analysis/deployment-probes/20260909-experiment-checks/flash-v3.json).

Preserved v3 observations: eight responses, 82,087 prompt tokens, 657 completions, 13.162 s summed client latency; five Actor responses returned offered schema-valid `execute_python`, zero reported reasoning tokens throughout. The five Actor request elapsed times were **2.354, 1.278, 1.835, 3.551, 2.032 s**, effective rates **45.4, 56.3, 42.5, 79.4, 38.9 tokens/s**. They are individual observations in an invalid run, not a sustained rate or matched comparison. No returned code was executed. [Raw response ledger](../../../analysis/deployment-probes/20260909-flash-next-feasibility/raw-v3/results.jsonl).

MTP counter deltas were **222 drafts, 665 proposed tokens, 441 accepted = 66.3% pooled acceptance**. There are no per-request snapshots or repetitions; draft tokens are not independent trials. V2's **32/32** accepted tokens over just **39 completion tokens** is likewise a tiny descriptive observation, not a stable rate. Both profiles were sampled, so describing v2 completions as deterministic is unsupported. No MTP-off arm establishes speed benefit.

Memory and readiness corrections matter: **410.056 s readiness + 103.938 s extraction ≈ 514.107 s setup**. The sampled all-run minimum free memory was 1259 MiB, while three post-readiness samples showed 5187/3027/3027 MiB. Neither is the instantaneous serving minimum; the allocator OOM shows sampling can miss a sharper trough. A nominal KV capacity quotient does not prove concurrent serving, 64k feasibility, or impossibility. The proposed “OOM skipped a tactic and made the chosen kernel slower” explanation is unproved; reclaim/retry also fits. Three cold JIT compilations during requests are directly recorded. Identity verification was fast-path/identity-only with zero payload files checked, not complete live weight-byte re-verification. [Independent analysis](../../../analysis/deployment-probes/20260909-flash-next-feasibility/INDEPENDENT-ANALYSIS-astra.md).

Coverage is independently inadequate for the live tail:

| Provider-reported prompt distribution | Count | Min | Median | Nearest-rank p90 | Max |
|---|---:|---:|---:|---:|---:|
| Five Flash fixtures | 5 | 10,559 | 17,578 | 21,257 | 21,257 |
| Later contingent-arm Actor calls combined | 136 | 12,664 | 21,605.5 | 31,534 | 38,644 |

Eight live counts exceed 32,768 prompt-only; eleven exceed it with a 1024-token output reservation. The largest fixture leaves **10,487 tokens after that reservation**, which says nothing about the omitted maximum. Exact requests must be retokenized with the candidate tokenizer/template before concluding which live calls Flash would reject; provider count differences are not exact compatibility proofs. These counts are workload-coverage evidence, not validation of the contingent-controller treatment itself. The original “first direct Actor token measurement” claim is false as a project-wide statement: September 6 capture already retained usage counts. [Coverage audit with raw source lines](../../../analysis/deployment-probes/20260909-experiment-checks/coverage.json).

V2 returned syntactically valid `{"key":4}` when the synthetic exact-output check expected `{"value":4}`; v3 reworded the prompt and passed. Ambiguous wording is plausible, but a single stochastic before/after does not prove the cause. Whitespace-only reasoning was a constructed test hazard, not an observed server defect. Requiring a tool when the request says `tool_choice:auto` over-implements the card. Removing that requirement did not affect v3, because all five Actor samples happened to return tools. These corrections belong in experimental interpretation even though OOM already invalidates both sessions.

## Repeated ideas and retained prior art

| Reappearing idea | What was already learned | What remains unearned |
|---|---|---|
| “MTP is fast, integrate it” | Synthetic +58.6%; real sampler crash; repaired long control −29.7%. | Net benefit on a matched relevant workload; high acceptance alone is insufficient. |
| “Batch requests to use the GPU” | Baseline-cost and Gregg measured actual server overlap and captured-set throughput. Successor publication can escape the decide semaphore. | Cross-game ownership/deadline/performance qualification; lock presence is neither proof of C1 nor proof batching is impossible. |
| “Fix the prefix cache / move volatile system text” | All 14 captured first system messages are identical. Exact-token prefix and recurrent checkpoint availability explain the selected pairs. | A cache correctness defect, a need to reorder prompts, or summary-boundary behavior. JSON cache-marker movement alone is not token divergence. [Corpus audit](../../../analysis/deployment-probes/20260907-prefix-cache/CORPUS.md) |
| “Raise the prefill chunk” | Completed 2048/4096 ABBA already tested the coupled graph/cache configuration and missed the bar. | A useful larger-chunk gain on the decision workload. Do not relabel this as untested. |
| “Reasoning must explain the slowness” | All 155 sustained q6 calls report zero reasoning tokens; repeated visible CodeAct calls consume time. | Hidden-reasoning cost as a cause of that run. |
| “A faster model solves the latency problem” | Actor decisions can consume 16–56 model calls before an action; Flash has no valid matched speed result. | Improvement in useful decisions or completed levels from a serving replacement. |
| “Green local tests / COMPLETE / HTTP 200 means pass” | Tokenizer API and assessor glob escaped fixtures; completed Flash responses coexist with OOM; captured HTTP 200 includes unoffered tools. | Card validity, executed-code correctness, and gameplay quality must each have evidence. |

## Strongest falsifiable lessons

1. **The relevant workload can reverse an engine selection.** MTP's sign reversal is measured; attributing it to sampling is falsified by retained greedy fixed requests. A causal explanation must discriminate the remaining context/content/runtime/order possibilities.
2. **Selection depends on the time origin and completion statistic.** Gregg's 95k C4 lowers makespan while raising median ready-to-completion. This counterexample defeats a general equation of throughput improvement with deadline improvement.
3. **A host lock's scope must include the full asynchronous request lifetime to imply C1.** The real-worker loopback counterexample produces two concurrent HTTP requests outside the claimed global serialization boundary.
4. **Prefix reuse requires matching recurrent state in this pinned hybrid path.** Observed 18,432/16,384 counts plus the 19,648 branch counterexample defeat a universal chunk-rounding rule. No evidence warrants bypassing the state guard.
5. **Instrumentation success and treatment success are separate.** Actor capture is useful despite zero actions; an MTP pilot that never reaches the Actor says nothing about that Actor's quality; schema-valid Flash tool strings are not executed decisions.
6. **An explicit invalidator outranks an attractive response table.** Flash's OOM invalidates both sessions under their unchanged card. The evidence does not become valid because cleanup and eight response gates succeeded.

## Retained configuration and unresolved boundaries

The measured retained baseline uses Qwen3.8-27B FP8, FP8 KV (`fp8_e4m3`), Triton attention, PyTorch sampling and no speculative algorithm. The prefix ABBA explicitly used `mem_fraction_static=.85`, retained chunk 2048, and production natural sampling temperature 1/top-k 20/top-p .95; its fixed controls deliberately reserve 256 outputs with EOS ignored. This statement is grounded in the [retained arm-0 server info](../../../analysis/deployment-probes/20260907-prefix-cache/raw/prefix-cache-20260907/arm-0-chunk-2048/server-info.json), not an assumption about today's mutable solver. FP8 weight representation and FP8 KV are different configuration choices. Neither implies MTP is enabled. The tested SGLang lineage is the pinned R8-L3B runtime; later Flash uses a different model, quantization and serving stack.

Still missing: a causal account of long-Actor MTP regression; full hybrid speculative rejection-state qualification; sustained real-game server overlap with ownership and deadlines; representative independent-game contexts; exact requests around summary boundaries; natural long-tail Flash retokenization; valid Flash feasibility and matched controls; and any demonstrated gameplay gain from these serving treatments. No inference-performance result here warrants the broader statement that the solver has never succeeded anywhere: these are bounded serving and selected ls20 observations, not the parent report's complete capability corpus.

## Checks performed for this appendix

Read-only source and ledger inspection recomputed the 153-call timing totals, prompt bands and turn-30 total; read both MTP fixed request samplers and first-control draft counters; checked all four long-successor cache counts; recomputed Gregg finite-set ready medians from raw request rows; and inspected Flash's allocator OOM alongside the eight-row response ledger. Published reassessment results are reported as prior checks, not rerun GPU or local qualification. Relative source links were checked for existence. No model launch, install, solver edit, board message or subdelegation was performed. Only this appendix was authored.
