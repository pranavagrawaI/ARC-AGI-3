# phase-a-qwen38-fp8-sglang-c1-r4

Question: Can the immutable Qwen3.8-27B-FP8 checkpoint and pinned SGLang route cold-load in a network-disabled Kaggle Phase A session on the RTX PRO 6000 and complete one visual Tycho-shaped native-tool request?

Prior runs: R1 is invalid due to generated Python syntax. R2 exposed omitted prerelease pins. R3 resolved those pins, then pip rejected the runtime dataset's deliberate dependency override: SGLang metadata requires `openai==2.6.1`, while the dataset lock and wheelhouse intentionally select `openai==3.5.0` under uv's workspace override.

Sole treatment change: install `uv==0.12.7` from the same offline wheelhouse and use uv's `--override` mechanism with the six overrides recorded by the dataset lock (`flashinfer-python`, `numba`, `compressed-tensors`, `outlines`, `opencv-python-headless`, and `openai`). Resolve only the SGLang route and its dependencies; do not install the unrelated full vLLM/JAX stack. Model input, runtime dataset and hashes, accelerator, server command, request, budgets, and acceptance rules remain fixed.

Prediction: the offline runtime installs; all 80 runtime checkpoint files match revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` reaches ready within 1,200 seconds; one ARC-like image request returns non-empty parsed reasoning and exactly one `take_action({"action":"LEFT"})` call.

Measure: `/kaggle/working/qwen38-fp8-phase-a-probe/result.json`, model/system/readiness/server/request/response evidence in that directory, and resolver/install/package evidence under `/kaggle/working/qwen38-fp8-phase-a-bootstrap/`.

Cap: one RTX PRO 6000; one server cold start; 1,200-second startup timeout; one inference request; 1,024 requested output tokens; 180-second request timeout; no ARC actions; no Phase B submission; no internet-enabled retry.

Stop: pass after the one validated tool call; abort at the first model-identity, runtime-provenance, GPU, install, load, CUDA/kernel, cache, readiness, HTTP, vision-result, reasoning-parser, or tool-parser failure.

Invalid if: internet is enabled; GPU, model, or runtime identity is unbound; attached inputs change; any model download occurs; more than one inference request occurs; or decisive raw evidence is missing.

Result map: pass → retain exact route as Stage 1 baseline and build frozen Tycho traces; model mismatch → replace only model attachment; runtime/install/load/kernel failure → repair or replace only runtime artifact; request/parser failure after readiness → repair only serving flags or payload; ambiguous or invalid → preserve and make no deployment-feasibility claim.
