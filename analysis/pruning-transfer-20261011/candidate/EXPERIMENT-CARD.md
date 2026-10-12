# prune-retain-candidate-20261011
Question: Does transferring retained REAP448 pruning and spending its memory savings on 18 streams improve gameplay over retained c14?
Posture: matched bundled comparison, conditional on successful transfer qualification.
Contrast: same Intel checkpoint/SGLang and c14 solver/context policy as control; target experts 512 -> 448 using the original recovered mask; active streams 14 -> 18, Mamba slots 84 -> 108, graphs include 14/16/18. Full512 drafter, shared expert, original atomic reduction, precision, prompts and sampler unchanged. Router bytes are checked before row selection on all 48 layers.
Prediction: approximately 7.5 GiB less target expert storage and higher aggregate throughput; gameplay may improve through capacity or regress through removed experts. Removing stored experts does not reduce top-k active experts (still 10).
Measure: output/benchmark.json total completed levels (primary), mean final_score (secondary), raw request logs and serve.log, target/draft receipts and source hashes. Use the identical estimator and validity checks frozen in control/EXPERIMENT-CARD.md.
Cap: one RTX PRO 6000, one launch, platform hard timeout 9000 s; 7254 s/game, all 25 public games, one pass. Entire investigation at most 21600 GPU-seconds including probe and control. No submission or checkpoint upload.
Stop: completed benchmark and harness teardown; source/shape/router/startup failure stops before gameplay; hard timeout contains outstanding work. No retry or fallback launch.
Invalid if: any paired validity condition in control/EXPERIMENT-CARD.md fails, target is not 448 experts in all 48 layers, draft is not 512, or routing/quantization metadata is inconsistent.
Result map: the exact paired rule in control/EXPERIMENT-CARD.md applies: require >=10% more levels, nondecreasing mean score, >=10% throughput gain and zero retraction log events to retain this bundle as the next local testing baseline. Negative or ambiguous results preserve c14. Never replace the retained mask based on this run or claim pruning alone caused the result.
Outputs: analysis/pruning-transfer-20261011/candidate/{launch-attempt.json,launch-response.json,status-history.jsonl,output/,collection.json}; paired RESULT.json and retention.json.
Authority: Pranav's 2026-10-11 request for a few tests with retention; launch only after valid probe and paired admission. No automatic repairs, retries, submissions or publication.

Amendment 2026-10-12: notebook rebuilt after probe 1 so its embedded pruning.py carries the drafter expert-map guard (probe2/EXPERIMENT-CARD.md) and smoke.py carries the misload floors. The guard defers every id below the target expert count to the stock wheel, so it is a no-op wrapper in the 512-expert control.
