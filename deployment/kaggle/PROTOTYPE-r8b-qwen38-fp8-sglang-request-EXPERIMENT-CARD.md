# R8-B — Qwen3.8 FP8 production request-path qualification

**Question:** Can the R8-A2-qualified Qwen3.8 FP8 substrate, with sampling explicitly resolved to PyTorch, accept one frozen visual request and return exactly one parser-accepted native `take_action({"action":"LEFT"})` tool call?

**Posture:** Execution for request-path qualification. Diagnosis only after a classified failure. Lifecycle remains a separately failed property.

**Predecessor:** R8-A2 passed initialization qualification and failed lifecycle qualification. Its immutable verdict is `analysis/deployment-probes/20260829T161921Z-qwen38-r8a2-kaggle-v8/VERDICT.md`, SHA-256 `e3238a206884f17e25e7a5d211af61d97ba42028c5e305ea2d8235c592bd49e8`. Its result SHA-256 is `9a829e2a75383d369fbf6826f5660c809ffd0d079dbf34c2a03ad6120f5216f7`; backend-facts SHA-256 is `319d00d5889e2f6fda93701315232eeab38496320430a7cf96abdd1c2179f7ba`; server-command SHA-256 is `30a8b9c56b583d6a0b81aab4c799382a3afd16f5eec53f98945ba7b20fa1a453`; shutdown SHA-256 is `d9c8083ae85f4e6e8690647f8bd727cd88dc8a616a39cc84cacd0d4894b97fd8`.

**Contrast:** Preserve the R8-A2-qualified model artifact and revision, model manifest, SGLang wheel and dependency lock, Kaggle RTX PRO 6000 / SM120 hardware request, offline installation, Triton attention path, graph/cache configuration, reasoning parser, tool parser, network policy, startup budget, and lifecycle code. R8-B makes one epistemic intervention: it explicitly binds `--sampling-backend pytorch`, verifies that attention actually resolves to `triton` and sampling actually resolves to `pytorch`, then sends one frozen representative request. R8-A2 did not exercise sampling or generation, so its dormant implicit FlashInfer sampler is not part of the qualified surface.

**Prediction:** The server again reaches readiness with attention resolved to `triton`; sampling resolves to `pytorch`; the exact 2,447-byte request receives HTTP 200; the response contains nonempty `reasoning_content`, `finish_reason=tool_calls`, and exactly one native `take_action` call whose parsed arguments equal `{"action":"LEFT"}`. The unchanged teardown may again require `SIGKILL`; that observation is reported separately and does not alter the R8-B request verdict.

**Measure:**

- probe source: `deployment/kaggle/PROTOTYPE-qwen3_8_phase_a_probe.py`, SHA-256 `88a18772929dfcd1613e34e2ff6d4df3ebd83d8cf5b2e7f6bc213ea5ca94d644`;
- notebook builder: `deployment/kaggle/scripts/build_phase_a_probe_notebook.py`, SHA-256 `6f67702c8dd9a8f4e39b947cc2da951006ff79f5c69cc68d710c54a74f4cc6eb`;
- model manifest: `deployment/kaggle/PROTOTYPE-qwen3_8_fp8-manifest.json`, SHA-256 `2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4`;
- frozen notebook: `deployment/kaggle/r8b-notebook/r8b_probe.ipynb`, SHA-256 `91b9565ada45573950a124ebf100cc872072ecdc883c81717f54eeb3599a2a0f`;
- kernel metadata: `deployment/kaggle/r8b-notebook/kernel-metadata.json`, SHA-256 `7a33c36c52b212eb9b535989300194e73cda4eb1c5a93ef9743cc66cee7cefa0`;
- model: `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- SGLang: `0.5.19.dev591+g8e04f66a7`, commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- SGLang wheel SHA-256: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- dependency-lock SHA-256: `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- exact served-model identifier: `/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`;
- request image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`;
- exact request: 2,447 bytes, SHA-256 `f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0`;
- raw execution artifacts: `probe-config.json`, `system-before.json`, `model-verification.json`, `server-command.json`, `server-environment.json`, `server.log`, `readiness.json`, `backend-facts.json`, `arc-like-probe.png`, `request.bin`, `request.json`, `request-identity.json`, `response.bin`, `response-transport.json`, `response.json` when decodable, `response-validation.json` when accepted, `result.json`, `shutdown.json`, `system-after.json`, and `events.log`.

**Cap:** Exactly one private Kaggle kernel push and execution under this R8-B identity after separate explicit user authorization. CLI and metadata must both request `NvidiaRtxPro6000`. Batch timeout: 1,800 seconds. GPU exposure: at most 0.50 GPU-hours. Startup timeout: 1,200 seconds. Request timeout: 180 seconds. Model requests: exactly 1 after readiness and backend-resolution checks. ARC actions: 0. Competition submissions: 0. Automatic retries: 0. Last verified quota before freeze: 0.25 hours used and 29.75 hours remaining of 30.00 hours; recheck immediately before any authorized push.

**Stop:**

- success: readiness passes; resolved attention is `triton`; resolved sampling is `pytorch`; the frozen request is sent once; HTTP 200 raw response bytes are preserved; and response validation accepts exactly one native `take_action({"action":"LEFT"})` call;
- futility: the server exits before readiness, startup exceeds 1,200 seconds, the single request exceeds 180 seconds, or any required identity/backend/request check fails;
- abort before model work: observed GPU is not RTX PRO 6000, internet is enabled, attached model/runtime identities fail verification, offline installation fails, quota or private access cannot be verified, or an output path already exists;
- after any failure: preserve and classify the exact evidence. Do not retry automatically. Do not repair lifecycle inside R8-B.

**Invalid if:** The pushed notebook or metadata digest differs from this card; either accelerator binding is omitted; the live GPU is not RTX PRO 6000; network access is enabled; model, manifest, runtime wheel, lock, graph/cache settings, Triton attention path, reasoning parser, tool parser, or lifecycle code differs from the frozen candidate; readiness is not reached; actual attention does not resolve to `triton`; actual sampling does not resolve to `pytorch`; request image, length, body digest, served-model identifier, tool schema, or request count differs; a redirect is followed; evidence is overwritten; or any automatic retry occurs. A startup or backend-resolution failure after adding the PyTorch binding is an invalid request test, not evidence about request execution.

**Result map:**

- startup or backend resolution fails before the frozen request can be sent → `R8-B INVALID`; classify `startup_invalid_after_sampling_binding` or the exact pre-request boundary; do not infer request behavior;
- readiness and backend resolution pass, but request transport fails without a generation/kernel marker → `request_serving_failure`; preserve transport error and server log;
- readiness and backend resolution pass, then server log records a CUDA, Triton, kernel, or traceback marker during the request → `generation_path_failure`;
- HTTP 200 body is not one decodable OpenAI-style response object → `generation_response_protocol_failure`;
- response semantics choose the wrong action or provide no evidence of the required visual relation → `multimodal_contract_failure`;
- response text expresses the correct `LEFT` semantics but contains no native tool-call structure → `tool_generation_failure`;
- tool markup/native structure is present but the production parser yields a malformed call, wrong function, invalid arguments, missing reasoning, or wrong finish reason → `parser_contract_failure`;
- resolved backends match, raw request/response bytes are preserved, and the production response parser accepts exactly one `take_action({"action":"LEFT"})` call → `R8-B PASS`; authorize only R8-L1;
- teardown again requires `SIGKILL` or fails to release GPU memory → separately retain `lifecycle FAIL`; do not change the R8-B request verdict;
- R8-B passes and shutdown is unexpectedly clean → retain both observations; still use R8-L1 or an explicit lifecycle disposition before S0 because R8-B did not change or qualify lifecycle behavior.

**Outputs:** Kaggle kernel `pranavagrawai/prototype-qwen3-8-fp8-phase-a-qualification` at the immutable version created by the separately authorized push; full pulled output and kernel log under a new timestamped `analysis/deployment-probes/` directory; an additive verdict that reports startup, visual request, native call, parser, and shutdown separately. This card is immutable once candidate review begins. It grants no launch permission.
