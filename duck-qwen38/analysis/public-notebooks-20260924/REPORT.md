# Public notebook source survey — 2026-09-24

## Scope and conclusion
Read-only investigation continuing `/tmp/handoff-duck-qwen38-20260924.md`. Retrieved a 100-entry competition notebook catalog sorted by scoreDescending, inspected five notebook/script sources and their attachment metadata, downloaded selected source/config files from two underlying solver bundles, and inspected Keith's public output. No competitor code or pickle was executed; pickle configuration was read with pickletools. No model run, push, or submission occurred.

The strongest distinct candidate in this sample is a model/serving change: Keith's Flash-Next NVFP4 stack. Several prominently named entries are copies or tiny variants of it. Another genuinely distinct family is Jakob's animation-aware Duck bundle used by Foysal. Notebook titles do not establish architecture, score, or active configuration.

## What each entry actually does

| Entry | Verified downloaded source behavior | Evidence |
|---|---|---|
| [Keith Flash-Next](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp) | Flash-Next NVFP4, ModelOpt FP4 weights/BF16 compute; custom FP8 PLE compatibility and CPU offload support; MTP 3, eight vLLM sequences, 28 Duck games, 32K context, 8K batched tokens, 5 GiB KV allocation, prefix caching disabled. 7,920 seconds/game, 900-second analyzer timeout, 60-second yield configured by serving setup. | `keithtyser/notebook.txt`, `keith-bundle/serving_setup.py`, public output log |
| [Wuliao Anim Base](https://www.kaggle.com/code/wuliao0/duck-qwen3-8-anim-base) | Downloaded code cells identical to Keith, same dataset/model refs. Title does not establish animation-aware behavior. | `verify.py`, metadata |
| [Chimgozirim Tuned](https://www.kaggle.com/code/chiakazirim/duck-qwen3-8-tuned) | Three source changes versus Keith: analyzer timeout 900→1200; gateway polling budget 600→300; accepted gateway response status <500→<250. Same attached dataset/model refs. | Diff of extracted notebook text |
| [Aman Hybrid REPL](https://www.kaggle.com/code/amanatar/arc-agi-3-hybrid-repl-agent) | Uses Keith's bundle/model. Sets n_passes=3 at line 222, overwritten to 1 at line 349 before execution. Request logging enabled. No separate hybrid solver implementation established in this script. | `amanatar/arc-agi-3-hybrid-repl-agent.py` |
| [Foysal LB-9](https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b) | Dense Qwen3.8-27B FP8, Jakob animation bundle. Pickle explicitly enables animation_awareness and hard_noop_guard. Animation metadata, transient regions, timeline and historical-animation queries extend observations; exact board/action guard blocks repeats recorded as unchanged, exempting animated actions. | `foysalemonshanto/notebook.txt`, `anim-bundle/benchmark.dis.txt`, downloaded prompts, solver, tool agent, animation and guard sources |

Keith's bundled base `prompts.py` is byte-identical to our current base prompt file. Differences against our tool agent and solver are preserved separately; our active flag-based prompt additions mean this is not proof that full runtime prompts are identical.

## Submission path versus public evaluation
Keith's notebook detects KAGGLE_IS_COMPETITION_RERUN. Public execution selects exactly 25 bundled offline games. Competition reruns obtain the live environment list through the gateway and use competition-mode Arcade. It forces one pass before bm.run. The placeholder submission.parquet is emitted only in the offline branch; it is not the source of competition performance. Model and source come from mounted attachments, restored deployment and benchmark objects, and serving setup. The full gateway/scorer implementation was not separately audited in this pass.

Foysal likewise switches to gateway games for competition execution; its 25-game selection is conditional on public mode. Public evaluation output and Kaggle competition score must remain separate measurements.

## Evidence of results
`keith-output/benchmark.json` records 25 games, one pass, 36 completed levels, 3,695 action-history entries, mean final_score 6.759036749169548. Its log agrees on game/action counts and configured serving profile. This is analysis of a published run, not our independent rerun. `verify.py` reproduces these totals and code comparisons.

The indexed [Keith notebook page](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp) reports competition score 3.38, best version 14. In [author comments](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp/comments?scriptVersionId=346399440), Keith attributes 2.36→3.38 to resubmission of the same version and reports throughput tuning from 114 to 236 tok/s. These are author statements; the causal throughput comparison is not independently established here.

The indexed [Foysal output page](https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b/output) reports 2.23, best version 6. “LB-9” must not be read as a score of 9.

## Limits that matter
- CLI pulls and dataset downloads retrieve current public content. Notebook metadata has model version refs but not pinned dataset version IDs. Identical attachment refs do not prove identical historical snapshots for each scored submission.
- Web-indexed scores can lag. Direct page reads for Wuliao, Chimgozirim and Aman failed; do not assign unverified scores from catalog ordering. The CSV returned no score values.
- Five sources are a bounded initial survey, not coverage of all open entries or private leading teams. The 100-entry catalog is preserved for expansion.
- A public result from one configuration does not isolate model quality, throughput, timeout, or run variance. Ranking clones does not establish efficacy of their small changes.
- Exact-board no-op detection still assumes visible board identity adequately represents relevant state. Animation exemptions reduce one failure mode; they do not prove that assumption universally.

## Judgment and next discriminating work
Compare distinct executable families before proposing another intervention. The first priority is to reconcile Keith's complete pinned runtime/model identity with our previous Flash-Next failure and current dense-model baseline. Our earlier MTP null on dense Qwen does not evaluate this different Flash-Next configuration. Historical submission-to-dataset linkage and comparable game-by-game action/time outcomes remain the next evidence gaps. Animation access is a separate observation change; do not bundle it with hard action blocking when testing its value.

Preserve the handoff's no-launch boundary. No experiment is authorized by this report.
