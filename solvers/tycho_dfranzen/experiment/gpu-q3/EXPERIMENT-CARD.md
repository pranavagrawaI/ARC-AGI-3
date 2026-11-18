# gpu-q3
Question: Does the locally qualified orchestrator hybrid execute correctly on the exact donor Kaggle serving stack?
Posture: Execution qualification; no efficacy claim.
Contrast: none.
Prediction: Offline dependencies install, FlashNext serves requests, hardened workers execute tools and acknowledge engine actions; raw traces reconcile with benchmark output.
Measure: downloaded notebook log, serve.log, qualification-start.json, tycho-dfranzen/candidate-config.json, benchmark.json, hybrid-games/*/worker-metadata.json, host-actions.jsonl, tool-trace.jsonl, decision-checkpoints, compute.json, execution-receipt.json. Inspect first successful and failed request if any, actual mode, role execution and isolation. A missing Builder invocation is unqualified, not proof the model learned without one.
Cap: 1 RTX PRO 6000; Kaggle hard timeout 3600 seconds including setup; at most 1 GPU-hour reserved; 3 games sb26/ls20/ar25, one pass, at most 16 actions/game and 600 seconds/game; 3 concurrent games; 12288 output tokens/call; no paid external endpoint; no submission. Total campaign cap 10 GPU-hours. No automatic retry.
Stop: Finish after the smoke or stop on installation/integrity/runtime failure; external timeout terminates at 3600 seconds. Abort interpretation if actual GPU count differs from one.
Invalid if: source or config hashes differ, wrong weights/hardware/game list, mode differs from orchestrator/control, offline installation fails, isolation/tool/action continuity fails, required raw receipts absent, or actions appear after cutoff. A timed-out model response is operational failure, not zero capability.
Result map: Valid execution -> prepare matched donor/hybrid comparison; failure -> preserve artifacts and diagnose before a separately identified retry; partial role coverage -> qualify the missing path before attributing mechanism effects; score changes -> no comparative conclusion.
Outputs: solvers/tycho_dfranzen/experiment/gpu-q3/output/ plus frozen kernel/, identity.json and launch/status receipts. No publication or competition submission.
Notebook SHA256: 5a194e9535bc90b902e90af6ddc6f60fe512f7400ac5dca10b65b4a314a40306
Parent notebook SHA256: b584e7cb9c6db37667ad39c81311a03c6964fc7d666c4b54ca1895a1fa60e653
Metadata SHA256: e42bf6577b9543444bb9ec56ea6b92f7907d9ad28b3e314cfcb9defbccfc79b7
Source manifest SHA256: 6c3b87c454b2651a21d25b198fe285c64246694dc66b6942d679e325109f70dd
