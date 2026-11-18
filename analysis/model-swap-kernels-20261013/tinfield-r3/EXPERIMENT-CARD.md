# model-swap-tinfield-smoke-20261013-v1
Question: Does the public Tinfield NVFP4 core run on the incumbent SGLang stack with the incumbent BF16 PLE and INT4 MTP drafter without gross gameplay breakage?
Posture: format qualification, then public gameplay smoke.
Contrast: none; historical c14 logs are a breakage reference, not a randomized selection experiment.
Prediction: server READY, finite token logprobs, median MTP acceptance >=2.0, all 25 games finish as won/gave_up, and >=44 levels at minute 30 on each game's clock (80% of the stated 55-level c14 reference).
Measure: exact-version output/serve.log, model-swap-finite-answer.json, model-swap-server-info.json, model-swap-assembly.json, benchmark.json, transcripts and event streams; levels_time.py unchanged; compare serving statistics over minutes 5-30 with the same window in c14 controls; report thinking tokens per executed action separately from generated tokens.
Cap: $0; one 6300-second GPU session (1.75 GPU-hours). Whole task <=12600 GPU seconds (3.5 hours); at most one additional run, only for a demonstrated pure infrastructure failure. No training. No competition submission call.
Stop: complete after 40 minutes of interactive play and normal teardown; stop at the platform cap or server/loader error. No automatic push retry. A model incompatibility or numerical failure is not infrastructure.
Invalid if: missing exact-version collection, no READY, missing or empty finite-answer evidence, wrong model/quantization/drafter/expert counts, missing c14 stream configuration, crashed game, unknown session accounting, or insufficient transcripts to measure the gate.
Result map: all gates pass -> queue this exact completed version for the next free hidden-draw day before its 20:30Z cutoff; failed gate -> preserve evidence and do not queue; ambiguity -> do not queue, diagnose read-only. Infrastructure failure -> retain v1 and use a new identity/card for one retry only within remaining task quota.
Outputs: analysis/model-swap-kernels-20261013/tinfield/output plus immutable launch identity, admission, attempt, response, status history and collection receipts.

The named base is the ledger c14 notebook, unchanged except target weights, the ModelOpt format selector, public weight assembly and interactive-only smoke controls. Its explicit TRUE_SUBMISSION branches are unchanged. All tokenizer/template/generation assets and the model config except quantization_config come from the incumbent. The INT4 group32 MTP drafter stays unchanged; target/drafter both have 512 experts. Tinfield's BF16 PLE was proved identical to Qwen via LFS hashes and Xet/ranged residual checks, so the public incumbent's unchanged BF16 table is reused. INT8 PLE and its custom vLLM runtime are unnecessary.

incumbent/check.py runs with --without b1: the exact named base lacks B1, while the mutable ledger requires its probation piece. Adding B1 violates the weights-only instruction. Board event 684 records this exception; thread 682 records D3's failed result. c14 and server-fail-fast are carried.

## r3 addendum (2026-10-13)
Same question, gates, caps and result map. Artifact: tinfield-r3/kernel/model-swap-tinfield-smoke-r3-20261013.ipynb (r2 + cell-4 fix dropping the core index unsharded ngram_embedding.weight entry whose file is not in the Kaggle core). One launch, no retry, 6300 s cap, task total <=12600 s. Authority: Pranav standing instruction 2026-10-13 relayed by the coordinator. Results under tinfield-r3/.
