# F0-R1 — Qwen3-VL fallback contract qualification

## Question

Can Kaggle model `qwen-lm/qwen-3-vl/transformers/8b-instruct-fp8/1`, bound offline to upstream `Qwen/Qwen3-VL-8B-Instruct-FP8` revision `9cdc6310a8cb770ce18efaf4e9935334512aee45`, cold-install the frozen vLLM runtime, start on an observed RTX PRO 6000 with FlashAttention language and vision backends, accept the primary route's exact visual request under its served-model alias, emit one parser-accepted native `take_action({"action":"LEFT"})` call with non-empty reasoning, and release its process group and GPU memory after shutdown?

This is a deployment-route qualification. It does not compare model quality with Qwen3.8, grant Game Judgment authority, execute an ARC action, score a task, create a QRM candidate, or authorize promotion.

## Attempt identity and exposure

- Attempt: `F0-R1`
- Kaggle kernel: `pranavagrawai/prototype-qwen3-vl-f0-fallback-contract-qualification`
- Kernel version: allocated by Kaggle after push
- Requested machine shape: `NvidiaRtxPro6000` in both metadata and CLI
- Internet: disabled
- Maximum GPU exposure: `0.50h`
- Model requests: exactly `1`
- Retries: `0`
- No automatic successor on failure

A pre-execution authentication or allocator failure is preserved under this attempt identity but does not test the model/runtime contract. Any retry requires a new attempt card and identity after classification.

## Frozen candidate

| Artifact | SHA-256 |
|---|---|
| `deployment/kaggle/PROTOTYPE-qwen3_vl_f0_probe.py` | `9ddcbdbca74cedd3187be36085498eb1e5278957a5903791e56036023eb371c6` |
| `deployment/kaggle/scripts/build_f0_probe_notebook.py` | `8d00414bf050d24d19ee89b69a467152ab7383e2d244c49f977881e3d28007ae` |
| `deployment/kaggle/f0-notebook/f0_probe.ipynb` | `244559fbc34e39c52332f9b7f9c816b5c51e18d018f75ffb09b99aef9dd224ca` |
| `deployment/kaggle/f0-notebook/kernel-metadata.json` | `ddd9b7acf9b660298f5b15cd8ea0bdbdb86c2578efc1b48a016095a9f2737dd6` |

The generated notebook embeds probe bytes exactly. All generated code cells compile. Metadata is private, network-disabled, binds `NvidiaRtxPro6000`, attaches the exact model version and runtime dataset, and attaches the ARC-AGI-3 competition source.

## Model identity

Before vLLM starts, the probe must:

1. discover exactly one `qwen3_vl` model mount;
2. require `git rev-parse HEAD == 9cdc6310a8cb770ce18efaf4e9935334512aee45`;
3. compare every runtime-relevant small working-tree file with `HEAD:<path>` bytes;
4. derive each shard's expected SHA-256 and size from the committed Git LFS pointer;
5. hash both mounted safetensor shards and require exact pointer equality;
6. require exactly two shards totaling `10,590,299,512` bytes;
7. require `Qwen3VLForConditionalGeneration`, `qwen3_vl`, FP8 `e4m3`, and dynamic activation quantization.

Kaggle's API reports the attached version at `10,606,952,507` bytes including Git metadata. Commit identity without LFS byte verification is insufficient.

## Runtime identity and installation

The attached runtime dataset is `mahdiseddigh/kaggle-t4-stack-2026-08-28`, version `1`.

| Runtime artifact | SHA-256 |
|---|---|
| `vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl` | `addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889` |
| `requirements.lock` | `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c` |

Installation uses only the mounted wheelhouse. The frozen override selects `flashinfer-python==0.6.17` against vLLM's stale exact metadata pin. After installation, the probe must observe:

- `vllm==0.28.0`
- `torch==2.13.0+cu130`
- `transformers==5.12.1`
- `flashinfer-python==0.6.17`
- `flash-attn-4==4.0.0b19`
- `triton==3.7.1`

## Frozen serving route

The server command must include:

```text
vllm serve <verified-mount>
--served-model-name Qwen/Qwen3.8-27B-FP8
--attention-backend FLASH_ATTN
--mm-encoder-attn-backend FLASH_ATTN
--enable-auto-tool-choice
--tool-call-parser hermes
--reasoning-parser qwen3
--host 127.0.0.1
--port 30000
```

`VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` are required. Hugging Face and Transformers offline modes are required.

## Frozen request

- Served-model alias: `Qwen/Qwen3.8-27B-FP8`
- Image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`
- Exact HTTP request-body SHA-256: `ffa037c93ef061d545eae5a780b194f1a76dc0fd24713783d82c01cfbde469f4`
- Exact body length: `2,405` bytes
- Endpoint: local `POST /v1/chat/completions`
- `tool_choice`: `required`
- Sampling: `temperature=0.0`, `max_tokens=128`
- Expected tool: exactly one native `take_action`
- Expected arguments: exactly `{"action":"LEFT"}`
- Expected parser evidence: `finish_reason == "tool_calls"`, one `tool_calls` item, and non-empty `reasoning_content`

The image object and serialized request bytes were compared directly with `PROTOTYPE-qwen3_8_phase_a_probe.py`; both are equal. Plain-text `LEFT`, the wrong action, missing reasoning, and multiple tool calls fail local negative controls.

## Pass conditions

F0-R1 passes only if all are true:

1. observed GPU is NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, with at least `90,000 MiB` total memory;
2. model commit, all committed runtime files, both LFS shard bytes, architecture, and quantization match;
3. frozen runtime wheel and lock match before install and installed package versions match afterward;
4. server becomes ready at `/v1/models` without a CUDA, kernel, or backend exception;
5. server exposes the primary served-model alias;
6. exactly one completion request is sent using the frozen bytes;
7. raw request and response bytes are preserved before parsing;
8. vLLM's Hermes and Qwen3 parsers produce one native `take_action(LEFT)` call with non-empty reasoning and no plain-text fallback;
9. a fresh process group exits after initial `SIGINT` without escalation;
10. no process remains in that group; and
11. GPU free memory returns to within `1,024 MiB` of the pre-server value within `60s`.

Any failed condition fails F0-R1. A completed Kaggle kernel is not a pass by itself.

## Failure classification

Classify the first violated boundary as one of:

- `invalid_pre_execution_authentication`
- `invalid_allocator_or_hardware`
- `model_mount_missing_or_ambiguous`
- `model_commit_mismatch`
- `model_worktree_mismatch`
- `model_lfs_blob_mismatch`
- `offline_runtime_installation_failure`
- `installed_runtime_identity_mismatch`
- `language_attention_backend_failure`
- `vision_attention_backend_failure`
- `model_initialization_failure:<type>`
- `request_transport_failure:<type>`
- `native_tool_parser_rejection`
- `reasoning_parser_rejection`
- `contract_response_mismatch`
- `unclean_process_group_shutdown`
- `gpu_memory_release_failure`
- `unrelated_failure:<type>`

Do not repair or rerun before preserving raw output and assigning one classification.

## Required evidence

Pull the exact completed kernel version, not an unqualified latest alias. Preserve:

- bootstrap result, install command, install log, package list, and runtime provenance;
- requested and observed metadata;
- system snapshots before and after;
- model verification including Git/LFS evidence;
- runtime verification;
- server command and environment;
- server log and readiness observations;
- exact image, request bytes, response bytes, parsed response, and validation;
- shutdown trace and process-group/memory samples;
- terminal result and failure traceback when present;
- pre/post quota snapshots;
- local candidate and readback hashes, with any source-readback transform stated explicitly.

## Licensed next move

A pass qualifies this exact F0 route as a retained deployment fallback and permits drafting its route manifest. It does not promote it, make it primary, or grant behavioral authority.

A failure preserves the boundary reached and closes F0-R1. No third GPU experiment may be launched from this card.
