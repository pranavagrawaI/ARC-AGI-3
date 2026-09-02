# phase-a-qwen38-fp8-sglang-c1-r5

Question: Can the immutable Qwen3.8-27B-FP8 checkpoint and pinned SGLang route cold-load in a network-disabled Kaggle Phase A session on the RTX PRO 6000 and complete one visual Tycho-shaped native-tool request?

Posture: bounded deployment-feasibility execution after accelerator-allocation diagnosis.

Contrast: R4 used the same notebook and inputs but Kaggle allocated a Tesla P100. R5 changes only the authenticated Kaggle server-side accelerator setting to RTX PRO 6000. The notebook SHA-256 remains `b6473786f7fa39211c2ee000788a79baab7f760ff7f291633bdf1c10e6141155`; kernel metadata SHA-256 remains `6c952766823cb8ee229612527618b87942a639d8f18cd4ba81e5b8d4362dd409`.

Prediction: the observed GPU contains `RTX PRO 6000`; the offline runtime installation succeeds; all 80 runtime checkpoint files match revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` reaches ready within 1,200 seconds; one ARC-like image request returns non-empty parsed reasoning and exactly one `take_action({"action":"LEFT"})` call.

Measure: `/kaggle/working/qwen38-fp8-phase-a-probe/result.json`, model/system/readiness/server/request/response evidence in that directory, and resolver/install/package evidence under `/kaggle/working/qwen38-fp8-phase-a-bootstrap/`.

Cap: one RTX PRO 6000 batch run; 30 minutes maximum interpreted run budget inside Kaggle's 9-hour notebook ceiling; one server cold start; 1,200-second startup timeout; one inference request; 1,024 requested output tokens; 180-second request timeout; no ARC actions; no Phase B submission; no internet-enabled retry. Kaggle reported a 30-hour weekly RTX PRO 6000 allowance with 29 hours remaining before this run. Reduced CPU availability is accepted as an expected condition and does not relax any timeout or pass criterion.

Stop: pass after the one validated tool call; abort at the first wrong-GPU, model-identity, runtime-provenance, install, load, CUDA/kernel, cache, readiness, HTTP, vision-result, reasoning-parser, or tool-parser failure. Do not start a second run from this card.

Invalid if: Kaggle does not allocate RTX PRO 6000; internet is enabled; model or runtime identity is unbound; attached inputs or notebook bytes differ from the fixed hashes; any model download occurs; more than one inference request occurs; or decisive raw evidence is missing.

Result map: pass → retain the exact route as the Stage 1 baseline and build frozen Tycho traces; wrong GPU → diagnose Kaggle allocation without changing the candidate; model mismatch → replace only model attachment; runtime/install/load/kernel failure → repair or replace only runtime artifact under a new card; request/parser failure after readiness → repair only serving flags or payload under a new card; ambiguous or invalid → preserve and make no deployment-feasibility claim.

Outputs: Kaggle kernel `pranavagrawai/prototype-qwen3-8-fp8-phase-a-qualification`, version created after this card; downloaded raw artifacts under a new immutable `analysis/deployment-probes/*-qwen38-phase-a-kaggle-v5/` directory.
