# Expected advantage and conditional ceilings

This is an offline interpretation, not a new experimental result or launch authorization. Measured deployment advantage remains zero; static M2 and M4 are unmeasured. The intended product is the incumbent solver with more useful work completed under the same time/GPU budget.

## Where MTP tuning could help

The target verifies a draft chain and emits the accepted prefix plus one bonus token. The pinned runtime metrics reporter accumulates correct_drafts+batch_size and divides by forward_count, so accept length includes that bonus. Throughput depends on accepted tokens divided by the full draft/verify/catch-up round cost. M2 trades fewer emitted tokens for less work; M4 trades more work for a longer accepted prefix. The target sampler/model and harness stay fixed under the Q7 configuration.

The measured invalid replay has M3 mean acceptance length A3=2.617907. Let qi denote the probability the first i draft tokens all survive verification. Then A3=1+q1+q2+q3, with q1>=q2>=q3>=q4>=0. If extending to M4 retains the distribution of the first3 proposals and prefixes, q4 <=q3 <=(A3-1)/3. Thus A4/A3-1 <=(A3-1)/(3*A3)=20.6005%. This is an upper bound on added token yield for the same sampled workload and unchanged first3 acceptance behavior. It is NOT a universal bound on MTP tuning, on M2 cost savings, or on deployed throughput. If M4 round cost does not fall, it also bounds the M3->M4 round-level speedup under those assumptions. Real scheduling, kernel shape, workload composition and finite-precision effects can invalidate those assumptions. We have no valid production M4 measurement.

For an illustrative model only, impose constant conditional acceptance p: A3=1+p+p^2+p^3. Solving gives p=.721565. Then A4=2.888990, +10.3549% token yield. If M4 costs5% more per round, net decode gain is5.0999%. If it costs10.3549% more, gain is zero. M2 model yield2.242221 is14.3507% lower than M3, so it needs at least14.3507% shorter rounds to break even. This constant-p model is not fitted/validated at each depth and the observed M1 accept length is different; use it as a sensitivity example, not a forecast.

## Conversion to useful gameplay

For a fixed work trace, let f be the fraction of critical-path time affected by decode acceleration and g be its speedup. Whole-trace speedup is 1/(1-f+f/g). f is not measured here. Cache fraction and GPU utilization do not establish f. At hypothetical f=.8, decode gains5%,10%,20% yield whole-trace gains3.96%,7.84%,15.38%. Extra work within a deadline then depends on scheduler/cache changes. Additional tokens become additional levels only if continuation is productive. There is no validated score conversion or guaranteed number of extra levels. Earlier board539's ~4levels/+10% speed was an observational elasticity heuristic; it is not a causal prediction. The historical deadline/fade confound is documented in analysis/lever-arena-20261002/LEVER-MAP.md.

A useful provisional engineering expectation is incremental speed, potentially a few percent net; zero gain remains plausible. Planning around0-10% decoder improvement would be an assumption-based prior, not a confidence interval or hard ceiling. A doubling has no support from these measurements.

## Larger original proposal

Expert-routing telemetry/offload/precision and concurrency changes were deferred from the frozen MTP card. Historical control analysis in board504#539 reported KV pool usage~66% and queue~0 at10streams, so freeing weight VRAM alone did not address an observed binding constraint. Expert frequency or phase concentration is not yet measured in this work. A useful routing intervention must reduce bytes transferred/kernel time on the critical path or demonstrably preserve more valuable context. If experts consume fraction fe of a step and their execution becomes2x faster, step speedup is1/(1-fe+fe/2): fe=.2 gives11.1%, fe=.5 gives33.3%. These are hypothetical sensitivities, not measured expert fractions or speedups. Offload/mixed precision can add transfer time or change quality and require separate qualification. A new serving engine has no measured basis yet.

## How far to take this

First diagnose occupancy and reload latency offline. Then a separately authorized small representative workload plus component timing should determine whether draft/verify costs leave recoverable headroom and whether M2/M4 improve it. Static tuning suffices if one setting wins consistently. Adaptive policies are justified only by measured phase/batch-dependent optima; current upstream SGLang documentation describes that mechanism but does not establish support in our pinned fork or a gain on ARC. Full model reloads are not evidence-producing work for the speed question.

The current4minute measurement has M3 window CV21.83%, larger than the conditional20.60% M4 yield ceiling and the plausible smaller gain. Together with invalid occupancy, this makes another identical sweep a poor discriminator. A replacement needs a predeclared measurement design that can resolve the desired effect, not a post-hoc relaxed adoption rule.

Final success means a valid serving gain, stable operation and retained output quality, followed by more useful gameplay progress at a matched deadline. If cheaper/faster execution just produces more stuck deliberation, the ARC advantage is zero. Retain M3 until a qualified replacement exists.

Sources: pinned metrics_reporter.py update_spec_metrics/report_decode_stats; frozen Q7-AUDIT.md and CARD.md; runtime events and sweep_state.json; board504#539; local LEVER-MAP.md; upstream https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx. Numerical sensitivities are retained in ceiling-estimates.json. No GPU used.
