# phase-a-qwen38-fp8-sglang-c1-r2

Question: Can the immutable Qwen3.8-27B-FP8 checkpoint and pinned SGLang route cold-load in a network-disabled Kaggle Phase A session on the RTX PRO 6000 and complete one visual Tycho-shaped native-tool request?

Prior run: R1 is invalid. Papermill stopped at 8.87 seconds because the notebook builder emitted an unterminated string literal in cell 2. No runtime installation, model validation, GPU load, or inference occurred.

Sole treatment change: escape generated newline literals correctly and compile every generated code cell locally before writing the notebook. Model input, runtime input, accelerator, server command, request, budgets, and acceptance rules remain fixed.

Prediction: all 80 runtime checkpoint files in the Kaggle snapshot match revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (the non-runtime `.gitattributes` transport file may be absent); SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` from Kaggle runtime dataset version 1 reaches ready within 1,200 seconds; one ARC-like image request returns non-empty parsed reasoning and exactly one `take_action({"action":"LEFT"})` call.

Measure: `/kaggle/working/qwen38-fp8-phase-a-probe/result.json`, `model-verification.json`, `system-before.json`, `system-after.json`, `readiness.json`, `server-command.json`, `server.log`, `request.json`, `response.json`, and `response-validation.json`; runtime installation evidence is under `/kaggle/working/qwen38-fp8-phase-a-bootstrap/`.

Cap: one RTX PRO 6000; one server cold start; 1,200-second startup timeout; one inference request; 1,024 requested output tokens; 180-second request timeout; no ARC actions; no Phase B submission; no internet-enabled retry in this run.

Stop: pass after the one validated tool call; abort at the first model-identity, runtime-provenance, GPU, install, load, CUDA/kernel, cache, readiness, HTTP, vision-result, reasoning-parser, or tool-parser failure.

Invalid if: internet is enabled; the GPU is not the observed RTX PRO 6000 tier; model or runtime identity is unbound; attached inputs change; any model download occurs; more than one inference request occurs; or decisive raw evidence is missing.

Result map: pass → retain exact sources/notebook/command/evidence as Stage 1 baseline, then build the frozen Tycho trace corpus; model mismatch → replace only model attachment; runtime/install/load/kernel failure → repair or replace only runtime artifact; request/parser failure after readiness → repair only serving flags or payload; ambiguous or invalid → preserve and make no deployment-feasibility claim.
