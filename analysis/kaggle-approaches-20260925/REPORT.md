# Working Kaggle approaches, checked 25 September 2026 IST

The most concrete reusable baseline found is Keith Tyser's Duck + Qwen3.8-Flash-Next NVFP4 serving stack. The much higher live leaderboard entries do not yet have an attributable current recipe in the sources inspected. Treat them as evidence that better solutions exist, not evidence for a guessed architecture.

This was read-only research: downloaded public source/data metadata and queried standings/submission receipts. No model calls, GPU jobs, submissions, solver edits, or trained-weight downloads.

## Live standings, not the stale search index

Retrieved through the authenticated official CLI at 2026-09-24 21:59 UTC (25 September IST). Raw response: `leaderboard-cli.txt`; timestamp: `retrieved-at.txt`.

| Position in returned leaderboard | Team | Public score |
|---|---|---:|
| 1 | Lord Han Solo | 19.40 |
| 2 | Tufa Labs | 18.81 |
| 3 | Yi-Chia Chen | 18.63 |
| 4 | NVARC3 | 16.07 |
| 5 | Daniel Franzen | 13.12 |
| 9 | Third Intelligence | 8.81 |
| 13 | mostik.ai | 7.51 |
| 15 | Son Pham & Mark Barney | 7.36 |

Source: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard

The search index still showed Mostik first at 7.51; it is stale. These are ongoing public standings, not final private results. The public API list of Lord Han Solo's competition notebooks returned none. Targeted searches did not recover a current recipe for the first five teams. Tufa's July open release does not identify its current 18.81 system. Mostik's separate latent-bridging demonstration does not identify its competition recipe.

## Reusable solutions and the strength of their evidence

**Duck + Flash-Next NVFP4 / MTP.** The indexed Kaggle notebook page reports 3.38 for V14. Its author reports an initial 2.36 and a later 3.38 on resubmission of that version, plus a local throughput change from 114 to 236 tokens/sec and public-demo score from 3.23 to 6.76. These are different measurements; the local score is not the Kaggle score. The run-to-run variation means 3.38 is not a proven expected value.

Notebook: https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp

Author's measurements: https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp/comments?scriptVersionId=346399440

The downloaded source and attached loader establish an implementable recipe: RadixArk Qwen3.8-Flash-Next NVFP4, BF16 compute, FP8 PLE CPU offload, a pinned vLLM runtime, native three-token MTP, 32K context, eight server sequences, 28 game threads, and a 7,920-second per-game cap. The notebook sets a 5 GiB KV budget, 8K batched tokens, and prefix caching off. These are that recipe's settings, not universal optima. Files: `notebooks/flash-mtp/`, `flash-bundle/serving_setup.py`, `flash-bundle/SOURCE_IDENTITY.json`. The loader was inspected, not executed.

**Duck + dense Qwen3.8-27B FP8.** FOYSAL's public V6 reports 2.23. It is a simpler baseline with a native Kaggle model input. Independently, our own official submission receipts confirm Qwen3.8 at 1.47 (56474012) and Qwen3.6 at 0.67 (56355127), both COMPLETE. The existing local report describes the same harness recipe; this is not a repeated causal estimate. Receipts are in `own-submission-check.json`.

Public notebook: https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b

Own report: /home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/qwen38-commit/RESULT.md

**Fork names are not mechanisms.** The downloaded executable cells of wuliao0's Anim Base, tantan0327's FlashNext AsIs, and Keith's baseline are identical. Chiakazirim's Tuned changes the analyzer timeout from 900 to 1,200 seconds and gateway waiting behavior. Aman Atar's Hybrid REPL includes a larger KV/batched-token budget and prefix caching enabled. These observations concern the downloaded versions; they do not identify code behind every historical best score. `notebook-source-comparison.json` records hashes. All five sources are preserved under `notebooks/`.

**Trajectory LoRA is an experiment, not an established winning recipe.** Manas Joshi reports a same-harness leaderboard improvement and releases data/adapter artifacts. Downloaded `train_meta.json` identifies only 9 conversations and 9,116 labeled tokens, rank 16, three epochs. The inspected model card lacks numerical evaluation results. This is concrete tooling with a positive author report, but insufficient evidence of a reliable gain.

Sources: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739047 and https://www.kaggle.com/datasets/justforgags/arc3-duck-lora-sft

**A useful negative training result.** Son/Mark's latest inspected seven-game comparison reports base/round3/round4W at 3/3/1 levels for the completed three-arm pass. The write-up-conditioned adapter did not help. Heavy timeout censoring, server restarts, and incomplete replication constrain interpretation. This does not explain their 7.36 Kaggle score or rule out better training data. Their detailed experiment supersedes broad prose claiming that self-training is already the source of competition gains.

Pinned source: https://github.com/sonpham-org/arc-3/blob/ff194c51c6012dfccbf8b0188d91e21c209985ef/docs/trace-findings/2026-09-20-arc3-round4-arm-w-gameplay-eval.md

**Conditional recovery and budget allocation have weaker supporting evidence.** In Fususu's experience-sharing thread, a participant reports a 2.86 submission rising to 3.48 on resubmission, and describes verified transition memory, one-action recovery, and conditional specialist use. A stopping allocator's 13.6% savings are a small-sample shadow estimate. Those mechanisms are candidates for tests, not separately demonstrated causes of the score.

Source: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739938

## Creative directions grounded in the working baseline

1. **Separate software reproduction from the research treatment.** Preserve the downloaded Flash-Next recipe as a complete isolated baseline. First establish that it serves and plays under our actual full-run budget. Then change one decision mechanism. This makes the baseline an asset even if a new idea fails.
2. **Distill decisions at forks.** At a failed student's exact observation prefix, ask a stronger teacher to select a distinguishing experiment using only that prefix. Execute and verify the candidate on permitted development environments, preserving failures and changed preconditions. Train on the decision and outcome, with matched alternatives. This targets the sb26 conditional-negative-evidence failure instead of imitating a fluent retrospective solution. A cheaper model must improve held-out gameplay to earn adoption.
3. **Allocate compute by value of the next experiment.** Separate the queue of games from the queue of model requests. Keep a minimum exploration allowance and revisit reserve, then allocate extra inference to games with testable hypotheses, reproducible near-solutions, or recent progress. Avoid stopping solely because no level has yet completed: our Qwen3.8 run contains late clears. The criterion should be expected score gained per compute, measured under equal total resources.
4. **Amortize established mechanics.** Use existing contingent execution to turn a verified local rule into a small helper and short checked action sequence. Return expensive reasoning at novelty or prediction failure. This does not require a complete renderer or universal simulator, and must beat the baseline after accounting for helper construction.

These are proposals, not claims about what the undisclosed leaders use. My ordering: reproduce Flash-Next first; run one narrow decision-improvement test next; pursue a training campaign only if the decision data measurably improve a held-out actor.

## Reproduction and source inventory

- `kaggle competitions leaderboard arc-prize-2026-arc-agi-3 --show --page-size 40 --format json`
- `kaggle kernels list --competition arc-prize-2026-arc-agi-3 --sort-by scoreDescending --page-size 20 --csv`
- Public sources pulled with `kaggle kernels pull OWNER/SLUG --metadata`; external notebooks were not executed.
- LoRA metadata and Flash loader fetched with single-file `kaggle datasets download ... -f FILE`.
- Son/Mark source pinned to `ff194c51c6012dfccbf8b0188d91e21c209985ef`, dated 2026-09-24. Tree and selected reports preserved.
- `FILE-HASHES.json` covers downloaded evidence and derived comparisons.
- Kaggle discussions/notebook scores cited above were recovered from indexed primary pages. Live browser control was unavailable; live CLI data supplies current standings and our submission receipts, not historical notebook score-version attribution.
