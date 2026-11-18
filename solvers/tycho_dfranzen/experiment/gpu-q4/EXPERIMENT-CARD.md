# gpu-q4
Question: Does the locally qualified orchestrator hybrid execute correctly on the exact donor Kaggle serving stack?
Posture: Execution qualification; no efficacy claim.
Contrast: none.
Prediction: Offline dependencies install, FlashNext serves requests, hardened workers execute tools and acknowledge engine actions; raw traces reconcile with benchmark output.
Measure: downloaded notebook log, serve.log, qualification-start.json, tycho-dfranzen/candidate-config.json, benchmark.json, hybrid-games/*/worker-metadata.json, host-actions.jsonl, tool-trace.jsonl, decision-checkpoints, compute.json, execution-receipt.json. Inspect first successful and failed request if any, actual mode, role execution and isolation. A missing Builder invocation is unqualified, not proof the model learned without one.
Cap: 1 RTX PRO 6000; Kaggle hard timeout 3600 seconds including setup; at most 1 GPU-hour reserved; 3 games sb26/ls20/ar25, one pass, at most 16 actions/game and 600 seconds/game; 3 concurrent games; 12288 output tokens/call; no paid external endpoint; no submission. Total campaign cap 10 GPU-hours. No automatic retry.
Stop: Finish after the smoke or stop on installation/integrity/runtime failure; external timeout terminates at 3600 seconds. Abort interpretation if actual GPU count differs from one.
Invalid if: source or config hashes differ, wrong weights/hardware/game list, mode differs from orchestrator/control, offline installation fails, isolation/tool/action continuity fails, required raw receipts absent, or actions appear after cutoff. A timed-out model response is operational failure, not zero capability.
Result map: Valid execution -> prepare matched donor/hybrid comparison; failure -> preserve artifacts and diagnose before a separately identified retry; partial role coverage -> qualify the missing path before attributing mechanism effects; score changes -> no comparative conclusion.
Outputs: solvers/tycho_dfranzen/experiment/gpu-q4/output/ plus frozen kernel/, identity.json and launch/status receipts. No publication or competition submission.
Notebook SHA256: 8dd1e95032939007d65b4eea5796b01c9238373d40bf3bba05319953cd4989ec
Parent notebook SHA256: 9d54a2a58a97d2055fa46074c8b0a75b785c759fd192c4e478c4082e2c19df01
Metadata SHA256: 2b335a7b258213edd5bf1df73bcf79969c3b911afd28e0f10467de37bfa80b2b
Source manifest SHA256: 2438fd2fd974c910285ceaf15cb6624643aafd906ba03284cba7fbdc2b84785c
