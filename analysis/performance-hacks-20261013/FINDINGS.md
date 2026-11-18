# ARC performance hacks: score and speed

Research snapshot, 2026-10-13. Target: more completed levels/score at the same budget, and faster model serving. Read current board, incumbent ledger, collected A5B result, constraint/Chen audits and primary papers/upstream source. No solver changes, model calls, GPU launches, training or submissions.

## Most useful new lead: separate active expert count from normalization

Collected A5B v1 (`138515992`) is COMPLETE but SMOKE_FAIL: top-5 target routing, unchanged top-10 drafter, W4A16 weights unchanged. Aggregate decode is 999.5 tok/s, +22.1%/+20.7% against the two c14 comparison windows. At 30 benchmark-clock minutes it completes 14 levels versus 44 for the first c14 control, failing the 39.6 no-harm floor. Its runtime routing receipt reports `norm_topk_prob=true`. Source: `analysis/model-swap-kernels-20261013/a5b/RESULT.json`, `EXPERIMENT-CARD.md`, `output/model-swap-routing.json`. This is a historical-control smoke, not a randomized replicated score comparison.

The training-free reference-mass paper separates two effects that plain top-k reduction bundles: omitted experts and amplification of survivors. Activate k1 expert outputs, but divide their router probabilities by the top-k2 probability mass. Proposed first operating point for our native top-10 model: k1=5, k2=10; retain original top-10 draft and shared experts. The already-tested arm is k1=k2=5. Compare these with the original k1=k2=10 at matched prefixes, context, concurrency and budgets. Do not replace missing experts with arbitrary outputs or simply disable normalization.

The paper reports small MMLU/GSM8K losses on other Qwen models under suitable reference mass; it does not test Flash-Next, long ARC tool trajectories or our quantized stack. It does not establish the cause of A5B's failure or promise its recovery. Its useful contribution is a distinct mechanism/control worth checking before retiring reduced active routing.

CPU arithmetic check executed this turn, with synthetic—not captured—router probabilities: top5 mass .25, top10 mass .375. Standard top5 normalization inflates each survivor weight by 1.5 versus top10-reference normalization, while both evaluate five expert outputs. This confirms the algebra only. Next CPU gate: implement against the exact packaged routing seam and compare IDs/weights with an independent reference; k1=k2=10 must reproduce native routing. GPU qualification remains separate.

Primary source: https://arxiv.org/html/2609.04575v1

## Ranked shortlist

| Priority | Trick | Existing evidence | First decisive check |
|---|---|---|---|
| 1 | Reduce active experts with preserved reference mass | Plain top5 is faster but badly fails gameplay; reference-mass correction is new here | CPU routing equivalence/weight oracle, then matched top10 versus top5/reference10 inference and tool-task quality; advance only with usable-output speed and gameplay no-harm |
| 2 | Tune MTP draft length to batch/context | Current NEXTN uses steps3/draft4; c14 median accept length about2.67. Upstream adaptive support varies steps with acceptance and batch size | Verify exact wheel support; paired steps1/2/3/4 at actual BS1/7/14 and short/long archived prefixes. Measure end-to-end throughput, memory and valid tool calls; acceptance alone is insufficient |
| 3 | Reserve finalization tokens to salvage cutoff reasoning | c14 has31 hard12,288-token cutoffs,30 with zero parsed tool calls; affected replies account for7.02% of response tokens | Matched checkpoint continuation, ordinary versus bounded artifact finalization at the same TOTAL token/time budget; compile/check code, legal tool receipt, prospective prediction and levels |
| 4 | Make planner resource failure explicit and preserve usable search work | Continuation audit has a learned model matching a winning29-action route while live BFS horizon was24; reference planner cleared it | Qualify general planner with FOUND/NO_PATH_WITHIN_HORIZON/RESOURCE_LIMIT/EXHAUSTED, then test on held-out learned models. Existing planner qualification is already underway; do not duplicate it |

MTP compatibility caution: current upstream documentation supports EAGLE/EAGLE3 topk1. Upstream argument resolver maps NEXTN to EAGLE, so the NEXTN spelling alone is not an incompatibility. Exact installed-wheel support and GDN/Mamba state correctness are unverified for adaptive switching. Begin with static sweeps if support is absent. Do not upgrade the serving stack just to obtain an unmeasured flag.

Primary MTP documentation: https://raw.githubusercontent.com/sgl-project/sglang/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx

Primary alias resolution: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/speculative_hook.py

Salvage/planner source evidence: `analysis/chen-transfer-20261013/FINDINGS.md`; `analysis/counterexample-continuations-v3-20261013/transcript-audit/findings.json`; `analysis/constraint-audit-20261013/FINDINGS.md`.

## Existing wins and low-priority directions

- c14's ~19.8% aggregate-decode gain is measured and retained, but a new score gain is not resolved. Faster tokens and more solved levels must be measured separately.
- REAP448+c18 has +10.7% aggregate decode and106 versus103 levels in the latest pair. This bundles pruning with concurrency and remains inconclusive for score. Do not present it as a fresh discovery.
- Solved-level action polishing has an idealized public-score ceiling of0.8–6.6% across audited cohorts, without accounting for extra completions/time spillovers. It is not the first big-score lead.
- Marlin non-atomic reduction has no useful timing win and failed numerical gates; changing another flag without a new bottleneck diagnosis is weak.
- Hazard-index scheduler's gain depends on unobserved deep-tail completions; capped worlds lose. Continue real tail investigation rather than promoting the simulated gain.
- B1 already preserves helpers across interruption; the prompt API contradiction is repaired in `m2_b1_fixed`, which is locally qualified but not gameplay-qualified. Both are existing work.

The recommendation is a narrow expert-reference-mass investigation first, with a separate MTP length screen for speed. Salvage offers a second route to more useful work per token. None of these new candidates has established ARC score benefit.
