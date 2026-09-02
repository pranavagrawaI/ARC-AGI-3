# phase-a-qwen38-fp8-sglang-c1-r1

Question: Can the immutable Qwen3.8-27B-FP8 checkpoint and pinned SGLang route cold-load in a network-disabled Kaggle Phase A session on the RTX PRO 6000 and complete one visual Tycho-shaped native-tool request?

Posture: bounded deployment-feasibility probe.

Contrast: none.

Prediction: all 80 runtime checkpoint files in the Kaggle snapshot match revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (the non-runtime `.gitattributes` transport file may be absent); SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` from Kaggle runtime dataset version 1 reaches ready within 1,200 seconds; one ARC-like image request returns non-empty parsed reasoning and exactly one `take_action({"action":"LEFT"})` call.

Measure: `/kaggle/working/qwen38-fp8-phase-a-probe/result.json`, `model-verification.json`, `system-before.json`, `system-after.json`, `readiness.json`, `server-command.json`, `server.log`, `request.json`, `response.json`, and `response-validation.json`.

Cap: one RTX PRO 6000; one server cold start; 1,200-second startup timeout; one inference request; 1,024 requested output tokens; 180-second request timeout; no ARC actions; no Phase B submission; no internet-enabled retry in this run.

Stop: pass after the one validated tool call; abort at the first model-identity, runtime-provenance, GPU, install, load, CUDA/kernel, cache, readiness, HTTP, vision-result, reasoning-parser, or tool-parser failure.

Invalid if: internet is enabled; the GPU is not the observed RTX PRO 6000 tier; model or runtime identity is unbound; attached inputs change after the card is fixed; any model download occurs during the run; more than one inference request occurs; raw request, response, server log, or result evidence is missing.

Result map: pass → retain the exact attached sources, notebook bytes, runtime command, and evidence as the Stage 1 baseline, then build the frozen Tycho trace corpus; model mismatch → rebuild the model attachment; runtime/install/load/kernel failure → repair or replace only the runtime artifact and issue a new run card; request/parser failure after readiness → repair the serving flags or client payload and issue a new run card; ambiguous or invalid → preserve the run and make no deployment-feasibility claim.

Outputs: `/kaggle/working/qwen38-fp8-phase-a-probe/`; after download, preserve under a new immutable directory in `analysis/deployment-probes/`.
