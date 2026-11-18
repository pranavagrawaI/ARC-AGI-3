# Arm P reviewed result

P passes the registered full-run rule. P1 completed 51 levels and P2 58: mean 54.5, +11.5 (+26.7%) over KF/C mean 43, exceeding the 51.5 threshold. Both remote notebooks report COMPLETE, all 25 games are present in each, all action events join to benchmark history, and final event scores agree with benchmark level totals. Exact remote notebook source was checked against the frozen P configuration.

The serving gain survived long contexts: median running requests were 8 in both P passes, including the second half, versus 3 in full KF/C runs. Median generation throughput while at least 20 requests were running or waiting was 384.1 and 385.4 tokens/s versus 248.1 and 245.85. Whole-run mean request latency was 97.89 and 95.48 seconds versus 145.26 and 146.88. P performed 5,495 and 5,515 actions, mean 5,505 versus 3,373 (+63.2%).

Gains were present in 15 games; 3 regressed; 7 tied on the mean of each pair. SB26 supplied +5 of the +11.5 mean-level gain; removing it leaves +6.5. G50T and SK48 cleared a level in both P runs and neither control. The clearest regressions are SP80 (P 0/0 versus KF/C 1/1) and TR87 (1/1 versus 2/5). Two passes establish passage of this precommitted decision rule, not absence of pruning-related quality loss on every game.

At 60 minutes, P had 41/38 levels versus KF/C 25/25; at 120 minutes it had 49/54 versus 41/41. The early lead persisted to the full horizon. P2 has one action just after the nominal 7,920-second cut: the time curve shows 5,514 actions there, whereas its final benchmark correctly totals 5,515. Its final level count is unchanged by that boundary.

Remaining serving pressure is still queueing: mean queue time was 63.30/61.09 seconds per completed request, about 64% of total request latency. Prefill was about 1.8 seconds and decode about 32 seconds. These are aggregate request measurements, not GPU idle-time fractions. Any next serving intervention should target this measured capacity constraint; increasing the 12-sequence limit alone is not supported by the observed median of 8 and approximately 92% occupied KV. No next intervention was launched.

Operational qualification: both P runs emitted the existing teardown exception after the final PUBLIC25_AUDIT. The teardown reports preserved required artifacts and metrics and closed the endpoint, but one vLLM GPU process remained at the bounded check, so shutdown_ok=false. The KF/C controls have the same flag. This is an explained post-game cleanup failure, recorded separately from the completed gameplay comparison; the runs are not operationally clean.

Decision: retain the unchanged P configuration as the local submission candidate; retain KF as fallback. No competition submission, rerun, or checkpoint upload was performed. The candidate is the shared configuration of P1/P2, not a claim that the better stochastic P2 trajectory will repeat. Remaining Kaggle GPU quota after collection: 8.48 hours, reset 2026-10-03 00:00 UTC.

Evidence: [reproducible report](REPORT.md), [raw-source hashes and metrics](assessment.json), [retention record](retention.json), [progress curves](progress.png). Raw outputs are in `/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/out-p1` and `out-p2`.
