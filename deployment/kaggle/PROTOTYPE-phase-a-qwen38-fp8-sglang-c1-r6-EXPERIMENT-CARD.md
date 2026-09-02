# phase-a-qwen38-fp8-sglang-c1-r6

Question: Can the immutable Qwen3.8-27B-FP8 checkpoint and pinned SGLang route cold-load in a network-disabled Kaggle Phase A session on the RTX PRO 6000 and complete one visual Tycho-shaped native-tool request?

Posture: bounded deployment-feasibility execution after identifying a difference between Kaggle UI draft settings and CLI-pushed batch settings.

Contrast: R5 used unchanged notebook bytes after RTX PRO 6000 was enabled in the authenticated editor, but a subsequent CLI `kernels push` still allocated `Tesla P100-PCIE-16GB`. R6 changes only the launch path: create the version through the authenticated editor's **Save Version / Save & Run All** flow while its RTX PRO 6000 setting is active. Do not use the CLI push path. Probe source, model/runtime inputs, request, budgets, and acceptance rules remain fixed.

Prediction: the observed GPU contains `RTX PRO 6000`; the offline runtime installation succeeds; all 80 runtime checkpoint files match revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` reaches ready within 1,200 seconds; one ARC-like image request returns non-empty parsed reasoning and exactly one `take_action({"action":"LEFT"})` call.

Measure: `/kaggle/working/qwen38-fp8-phase-a-probe/result.json`, model/system/readiness/server/request/response evidence in that directory, and resolver/install/package evidence under `/kaggle/working/qwen38-fp8-phase-a-bootstrap/`.

Cap: one UI-started RTX PRO 6000 batch run; 30 minutes maximum interpreted run budget; one server cold start; 1,200-second startup timeout; one inference request; 1,024 requested output tokens; 180-second request timeout; no ARC actions; no Phase B submission; no internet-enabled retry. The last quota observation before R5 was 29 of 30 weekly RTX hours remaining; R5 used a P100 and therefore does not establish any RTX quota debit. Reduced CPU availability is accepted and does not relax pass criteria.

Stop: pass after the one validated tool call; abort at the first wrong-GPU, model-identity, runtime-provenance, install, load, CUDA/kernel, cache, readiness, HTTP, vision-result, reasoning-parser, or tool-parser failure. Do not start another run from this card.

Invalid if: the save dialog does not show or preserve RTX PRO 6000; Kaggle allocates another GPU; internet is enabled; model or runtime identity is unbound; probe source or attached inputs change; any model download occurs; more than one inference request occurs; or decisive raw evidence is missing.

Result map: pass → retain exact route as the Stage 1 baseline and build frozen Tycho traces; wrong GPU → escalate the allocator/UI discrepancy without changing the candidate; model mismatch → replace only model attachment; runtime/install/load/kernel failure → repair or replace only runtime artifact under a new card; request/parser failure after readiness → repair only serving flags or payload under a new card; ambiguous or invalid → preserve and make no deployment-feasibility claim.

Outputs: the next Kaggle version created through the authenticated editor; downloaded raw artifacts under a new immutable `analysis/deployment-probes/*-qwen38-phase-a-kaggle-v6/` directory.
