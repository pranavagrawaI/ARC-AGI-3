# R8-B2 — Qwen3.8 FP8 warmup-complete request qualification

**Question:** After the pinned SGLang runtime completes its automatic VLM warmup, can the R8-A2-qualified Qwen3.8 FP8 route with PyTorch sampling accept exactly one frozen probe-owned visual request and return exactly one parser-accepted native `take_action({"action":"LEFT"})` tool call?

**Posture:** Execution for request-path qualification. Diagnosis only after a classified failure. Lifecycle remains a separate property.

**Predecessor:** R8-B version 9 produced a successful diagnostic target request but was invalid under its exact-one-total-model-request card because `/v1/models` returned while SGLang's automatic warmup thread was still running. The immutable verdict is `analysis/deployment-probes/20260829T185525Z-qwen38-r8b-kaggle-v9/VERDICT.md`, SHA-256 `545c4e165a65509b9387a2538e2ec3939be733b11ebbe1fc0acf8ca34e30a2f7`. Its result SHA-256 is `2b29abe0680df974708ea86eb567b9e8c137fc869868b22b44f94e54c16959c8`; server-log SHA-256 is `9c046e9723c3127eb5b8abfcfdaab8f620c2976a84df73142c12a8b895c10b48`; and preserved pinned SGLang `http_server.py` SHA-256 is `056422af61f51573256560509d42ed4f23151364158474aff1c012cce95c880b`.

**Contrast:** Preserve the R8-B model artifact and revision, model manifest, SGLang wheel and dependency lock, Kaggle RTX PRO 6000 / SM120 hardware request, offline installation, Triton attention, PyTorch sampling, graph/cache configuration, reasoning parser, tool parser, frozen visual request, network policy, total startup budget, request timeout, and lifecycle code. Change only the readiness and request-count contract:

1. retain `/v1/models` as an early model-API observation;
2. do not send the probe request until the pinned runtime logs exactly one successful VLM warmup completion access, exactly one successful `/freeze_gc` access, and exactly one `The server is fired up and ready to roll!` marker;
3. require exactly one additional successful `/v1/chat/completions` access after sending the frozen probe request;
4. preserve the pre-request warmup facts and post-request count facts as separate artifacts.

The production warmup remains enabled. R8-B2 does not change lifecycle behavior.

**Prediction:** The early model API responds; SGLang completes one runtime-owned VLM warmup inside the existing 1,200-second startup budget; resolved attention remains `triton`; resolved sampling remains `pytorch`; the exact 2,447-byte probe-owned request receives HTTP 200; the response contains nonempty `reasoning_content`, `finish_reason=tool_calls`, and exactly one native `take_action` call whose parsed arguments equal `{"action":"LEFT"}`. The complete log contains exactly two chat-completion accesses and both succeed: one runtime-owned warmup before the ready marker and one probe-owned request after it. Lifecycle is recorded but does not change the request verdict.

**Measure:**

- probe source: `deployment/kaggle/PROTOTYPE-qwen3_8_phase_a_probe.py`, SHA-256 `caf650598a9a15b290efe7cda710947a2d4b3ce092a5999c2019c46b5e14ad2e`;
- notebook builder: `deployment/kaggle/scripts/build_phase_a_probe_notebook.py`, SHA-256 `e96218af6adca1c6bcdfa0f52dd14c64777d8d20fa6dde45848e126168b8c65f`;
- model manifest: `deployment/kaggle/PROTOTYPE-qwen3_8_fp8-manifest.json`, SHA-256 `2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4`;
- frozen notebook: `deployment/kaggle/r8b2-notebook/r8b2_probe.ipynb`, SHA-256 `5abc0ca07ed0d481c7316f09848cababba4b43a707a1d108c50d78fb94703734`;
- kernel metadata: `deployment/kaggle/r8b2-notebook/kernel-metadata.json`, SHA-256 `2fd13b8dfbdcedf77ee885b6df75b41bcbcf6e0fb84808c74854bdd119ed9d72`;
- inherited frozen R8-B notebook: `deployment/kaggle/r8b-notebook/r8b_probe.ipynb`, SHA-256 `91b9565ada45573950a124ebf100cc872072ecdc883c81717f54eeb3599a2a0f`;
- inherited frozen R8-B metadata: `deployment/kaggle/r8b-notebook/kernel-metadata.json`, SHA-256 `7a33c36c52b212eb9b535989300194e73cda4eb1c5a93ef9743cc66cee7cefa0`;
- model: `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- SGLang: `0.5.19.dev591+g8e04f66a7`, commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`;
- SGLang wheel SHA-256: `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`;
- dependency-lock SHA-256: `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- exact served-model identifier: `/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`;
- request image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`;
- exact probe-owned request: 2,447 bytes, SHA-256 `f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0`;
- required runtime evidence: `model-api-readiness.json`, `warmup-facts.json`, `readiness.json`, `backend-facts.json`, `request-count-facts.json`, exact `request.bin`, exact `response.bin`, `response-transport.json`, `response.json`, `response-validation.json`, `result.json`, `server.log`, and the unchanged lifecycle artifacts.

**Cap:** Exactly one private Kaggle kernel push and execution under this R8-B2 identity after separate explicit user authorization bound to the final accepted bytes. CLI and metadata must both request `NvidiaRtxPro6000`. Batch timeout: 1,800 seconds. GPU exposure: at most 0.50 GPU-hours. Total model-API plus runtime-warmup startup timeout: 1,200 seconds. Probe-request timeout: 180 seconds. Runtime-owned warmup requests: exactly 1 and successful. Probe-owned representative requests: exactly 1 and successful. Total chat-completion accesses: exactly 2, both successful. ARC actions: 0. Competition submissions: 0. Automatic retries: 0. Last verified quota before this freeze: 0.36 hours used and 29.64 hours remaining of 30.00 hours; recheck immediately before any authorized push.

**Stop:**

- success: the early model API and warmup-complete boundaries pass; pre-request evidence shows exactly one successful runtime warmup, one successful GC freeze, and one ready marker; resolved backends are `triton` and `pytorch`; the frozen probe request is sent once; HTTP 200 raw response bytes are preserved; response validation accepts exactly one native `take_action({"action":"LEFT"})`; and post-request evidence shows exactly one additional successful completion access;
- futility: the server exits before either readiness boundary, total startup exceeds 1,200 seconds, the single probe request exceeds 180 seconds, or any required identity/backend/request-count check fails;
- abort before model work: observed GPU is not RTX PRO 6000, internet is enabled, attached model/runtime identities fail verification, offline installation fails, quota or private access cannot be verified, remote version 9 is no longer the predecessor, the output slug differs, or an output path already exists;
- after any failure: preserve and classify the exact evidence. Do not retry automatically. Do not repair lifecycle inside R8-B2.

**Invalid if:** The pushed notebook or metadata digest differs from this card; either accelerator binding is omitted; the live GPU is not RTX PRO 6000; network access is enabled; model, manifest, runtime wheel, lock, graph/cache settings, Triton attention, PyTorch sampling, reasoning parser, tool parser, request bytes, or lifecycle code differs from the frozen candidate; the current remote predecessor is not version 9 at `pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path`; the metadata ID and unchanged title no longer resolve to that slug; `/v1/models` does not identify the exact served model; the pre-request log does not contain exactly one warmup access that succeeded, one successful `/freeze_gc` access, and one ready marker; actual attention does not resolve to `triton`; actual sampling does not resolve to `pytorch`; request image, length, body digest, served-model identifier, tool schema, or probe-owned request count differs; total chat-completion accesses differ from two or either access is unsuccessful; a redirect is followed; evidence is overwritten; or any automatic retry occurs.

**Result map:**

- model API fails or runtime warmup does not finish inside the shared startup budget → `R8-B2 INVALID`; preserve the earlier boundary and classify the exact startup failure;
- warmup finishes but pre-request warmup/GC/ready counts differ → `R8-B2 INVALID: request_count_contract_invalid`; send no probe request;
- warmup and backend resolution pass, but request transport fails without a generation/kernel marker → `request_serving_failure`; preserve transport error and server log;
- warmup and backend resolution pass, then the server logs a CUDA, Triton, kernel, or request-time traceback marker → `generation_path_failure`;
- HTTP 200 body is not one decodable OpenAI-style response object → `generation_response_protocol_failure`;
- response semantics choose the wrong action or provide no evidence of the required visual relation → `multimodal_contract_failure`;
- response text expresses the correct `LEFT` semantics but contains no native tool-call structure → `tool_generation_failure`;
- tool markup/native structure is present but the parser yields a malformed call, wrong function, invalid arguments, missing reasoning, or wrong finish reason → `parser_contract_failure`;
- the response parses correctly but post-request counts differ from one warmup plus one probe request → `R8-B2 INVALID: request_count_contract_invalid`; retain the successful response as diagnostic evidence only;
- resolved backends match, warmup/request counts close exactly, raw request/response bytes are preserved, and the response parser accepts exactly one `take_action({"action":"LEFT"})` → `R8-B2 PASS`; authorize only R8-L1;
- lifecycle observations are reported separately and do not alter the R8-B2 request verdict. Even a clean parent exit and GPU-memory return do not replace R8-L1's process-tree and teardown-log requirements.

**Outputs:** Existing private Kaggle kernel `pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path` at the new immutable version created only by a separately authorized push; full pulled output and kernel log under a new timestamped `analysis/deployment-probes/` directory; an additive verdict reporting early model API, runtime warmup, resolved backends, probe request, native call, parser, request counts, control-plane identity, and lifecycle separately. This card is immutable once exact-byte review begins. It grants no launch permission.
