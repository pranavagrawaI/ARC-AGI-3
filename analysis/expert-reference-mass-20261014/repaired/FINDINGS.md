# Valid rerun: five experts decode faster but lose gameplay performance

The repaired screen completed successfully. **Keep native top10. Active5/reference10 is a NEGATIVE_QUALITY_SCREEN: +21.13% aggregate decode, but6 versus17 completed levels at exactly900 seconds (−64.71%).** The normalization correction did not preserve useful gameplay performance in this short screen. No candidate promotion, additional GPU launch or competition submission followed.

## Observed outcome

One private RTX PRO6000 session, kernel `pranavagrawai/expert-reference-mass-screen-r2-20261014`, id138721881, version1. Control first, candidate second; same25 public games, Intel W4A16 weights, native top10 MTP and frozen c14 serving/prompt settings. One stochastic replicate,15-minute fresh-game horizon per arm. Notebook containment reports2923.03s total including setup/startups/probes/play/shutdown (48.72min), below3600s. Control server reached READY after493s; candidate after352s.

| Measurement | Native10 control | Active5/reference10 |
|---|---:|---:|
| Completed levels at exactly900s | 17 | 6 |
| Final shutdown levels | 17 | 6 |
| Aggregate decode, minutes2–14 | 831.84 tok/s | 1,007.61 tok/s |
| Per-stream decode, same window | 62.11 tok/s | 73.42 tok/s |
| Median MTP acceptance length | 2.59 | 2.49 |
| Median running streams | 14 | 14 |
| Mean final-snapshot public score | 2.9282 | 0.8102 |

The candidate loses11 levels; level ratio0.35294, decode ratio1.21130. Final scores are shutdown snapshots. The primary level metric uses game-start offsets plus action-history cumulative wallclock and reconciles with action-event scores. Neither arm added another completed level after900s.

## Validity and repair verification

All20 readout gates pass. All25 initial boards match; all50 initial events are fresh score0/level1/action0. Serving configurations match:14 requests,102400 context,84 Mamba slots, AutoRound/BF16 target compute, FP8 KV, MTP steps3/draft4/topk1. No crashed games or retractions. All25 game states per arm are intentionally cancelled by the soft deadline; this is the declared shutdown behavior, not a crash. Both full-play exposures completed.

Gameplay child PIDs are1206 and3626, with parent66. Both gates were absent before play. Actual clocks are control[1270.6695,2169.7701] and candidate[2619.2477,3518.5336], fresh after their respective child initialization and bounded by900s. Candidate starts after the control deadline; there is no reused gate clock. The separate-process lifecycle repair passed on the real Kaggle path, beyond the earlier CPU reproduction.

The saved GPU unittest stderr records **Ran7 tests / OK**, no skips; stdout contains the explicit changed-input CUDA graph success marker; JSON returncode0. Both runtime source installations bind the same exact wheel/original module/helper hashes. Candidate markers cover all48 target layers active5/reference10 and unchanged top10 draft. Both serving logs identify AutoRound target `GPTQMarlinMoEMethod`, with deferred TRTLLM routing disabled. The exact wheel's STANDARD/native dispatcher calls the nonnull custom callback. This is source/backend evidence, not a per-token activation capture.

Collected229 selected raw receipts from433 listed outputs, bound accepted id/version before and after collection; all229 local sizes/SHA256 rechecks match. Remote notebook cells match the sealed source, and the card hash is unchanged. Raw output is excluded from Git checkpoints; collection/hash receipts are tracked. Movies/prompts/solver_analysis remain outside the collection scope.

## Why faster tokens did not help

Full-phase response diagnostics below include shutdown drain, rather than an exact900s response window:

| Diagnostic | Control | Candidate |
|---|---:|---:|
| Requests / completed response receipts | 482 /465 | 367 /349 |
| Tokens in completed responses | 698,779 | 892,018 |
| Mean completion tokens per reply | 1,502.75 | 2,555.93 |
| Median completion tokens per reply | 993 | 1,990 |
| Median reported reasoning tokens per reply | 772 | 1,728 |
| Recorded output-length cutoffs | 2 | 2 |

Candidate completed responses are24.95% fewer, despite27.65% more recorded completion tokens. Median reply length doubles; median reported reasoning length more than doubles. The increase is broader than a few long outliers: the75th percentile rises from1773 to3720.5 tokens. Among23 pairs of identical first-request messages, initial-reply medians are113 versus198 tokens; the mean is skewed by one6968-token candidate reply. Do not use the mean as a typical first-reply length.

This supports a concrete symptom: worse conversion of tokens into interactions and completed levels. The old excess-cutoff pattern did not reproduce; both arms have two recorded length cutoffs. All other completed replies report `tool_calls`. The deduplicated logged Python result census has433 versus310 result signatures and no nonzero returncodes; that is partial log coverage, not proof of error-free reasoning or complete tool-call coverage. Seventeen/control and eighteen/candidate attempted requests lack completed response receipts, so their final usage/finish reasons are not included.

The clean clock and test receipts exclude the two known v1 validity faults here. They do not separate missing expert contributions from reduced routed-branch magnitude: active5/reference10 changes both relative to native10. Fixed order, stochastic sampling, short exposure and diverging later game states remain limitations. This result supports rejecting this configuration, not a theorem against every reduced-expert method.

## What we can use

Retain native10 as the solver. Retain the repaired isolation, graph qualification, containment and readout as a reusable comparison harness. Preserve v1 as INVALID and this successor as a valid negative screen.

Shared premise under challenge: **halving active experts leaves enough useful computation for the decode gain to compensate for any quality loss.** Historical plain-top5 and this reference-mass successor both badly miss their gameplay gates. Correct normalization is insufficient to rescue this operating point. The fresh clocks, equal cutoffs and longer-reply distribution weaken scheduling/qualification/excess-cutoff explanations. Missing contributions versus reduced branch magnitude remain unresolved; do not immediately spend the remaining quota on an arbitrary expert-count sweep.

The next practical bet is a bounded static MTP-depth timing/qualification probe on the retained10-expert target and real short/long prefixes at c14 batch sizes. The observed acceptance length is2.59 with steps3/draft4; test whether less draft/verification work improves end-to-end throughput. This is a hypothesis, not an established gain or a guarantee of gameplay equivalence. Verify exact-wheel/Mamba/graph behavior first, measure useful completed outputs, and require separate authorization before another GPU run. If expert reduction is revisited, first measure omitted router mass and contribution error on representative inputs to distinguish the capacity and magnitude explanations.

Account readback after collection:27.5311h used/reserved,2.4689h remaining, no active sessions. This is an account snapshot, not an isolated billing receipt. No extra budget was consumed after the one accepted rerun.

## Reproduction

```bash
python3 -B analysis/expert-reference-mass-20261014/repaired/readout.py
python3 -B analysis/expert-reference-mass-20261014/repaired/diagnostics.py
python3 -B analysis/expert-reference-mass-20261014/repaired/test_readout.py
```

See `RESULT.json`, `collection.json`, `launch-identity.json`, `EXPERIMENT-CARD.md`, `evidence/conversion-diagnostics.json`, `evidence/response-length-distribution.json` and raw `output/reference-mass/`. The collector and monitor never launch GPU work; the launch command is one-attempt guarded and refuses a second attempt. The original v1 readout's hardcoded validity flags are not used by this successor.
