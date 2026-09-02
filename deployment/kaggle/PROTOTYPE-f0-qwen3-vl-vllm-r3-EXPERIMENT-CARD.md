# F0-R3 — SaveKernel identity correction

## Predecessor

F0-R2 was rejected before execution by Kaggle `SaveKernel` with HTTP 400:

- result: `analysis/deployment-probes/20260829T170704Z-qwen3vl-f0-r2-savekernel-400/result.json`
- rejected-byte archive: `analysis/deployment-probes/20260829T170704Z-qwen3vl-f0-r2-savekernel-400/rejected-candidate.tar.gz`
- archive SHA-256: `61d85f190003158590dbb81ca936146a550a2c6ded6687b830826b134db753f7`

F0-R2 created no remote kernel, consumed no GPU quota, installed no runtime, loaded no model, and sent no generation request.

## Supported diagnosis

F0-R2's title and slug were both 56 characters. Kaggle kernel titles are limited to 50 characters. The current client forwards metadata `title` as `new_title` but checks only a five-character minimum, leaving the server to reject the excessive title with HTTP 400.

This is a sufficient cause, not proof that every other metadata field will be accepted. R3 may reveal another latent control-plane defect.

The current client source also establishes that `--timeout 1800` becomes `session_timeout_seconds=1800`; it is a kernel-runtime ceiling.

## R3 question

Does shortening only the attempt/remote identity cross `SaveKernel`, and—if it does—can the otherwise unchanged F0 contract complete on the exact offline Kaggle substrate?

## Only intended R2 → R3 changes

- attempt label: `F0-R2` → `F0-R3`;
- output evidence paths: `f0-r2` → `f0-r3`;
- title: `PROTOTYPE Qwen3-VL F0 R2 fallback contract qualification` → `PROTOTYPE Qwen3 VL F0 R3`;
- kernel id: `pranavagrawai/prototype-qwen3-vl-f0-r2-fallback-contract-qualification` → `pranavagrawai/prototype-qwen3-vl-f0-r3`;
- notebook heading and default output directory follow the new attempt identity.

Kaggle's own `slugify` resolves the new 24-character title exactly to `prototype-qwen3-vl-f0-r3`.

## Frozen R3 bytes

| Artifact | SHA-256 |
|---|---|
| `deployment/kaggle/PROTOTYPE-qwen3_vl_f0_probe.py` | `5e9664558bed1ea03be7de6ecf413f2cfeda96108904e6c3b48d0f285294927b` |
| `deployment/kaggle/scripts/build_f0_probe_notebook.py` | `b6f4594fabed1d0225fefbf23657a206bfb3e8a5d827fdb7aa4b55b8f6b0bf28` |
| `deployment/kaggle/f0-r3-notebook/f0_probe.ipynb` | `c5e1233ee1637ec4e1a40f2cab9c15e09424a3ffbefc30d8283105dc0c808e31` |
| `deployment/kaggle/f0-r3-notebook/kernel-metadata.json` | `3640df1ccc361a1858f94dbd6565f2664b373540aa102fe5c340a2763ab41b29` |

## Unchanged F0 contract

- model source: `qwen-lm/qwen-3-vl/transformers/8b-instruct-fp8/1`;
- upstream revision: `9cdc6310a8cb770ce18efaf4e9935334512aee45`;
- vLLM: `0.28.0`, wheel SHA-256 `addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889`;
- source lock SHA-256 `d473b6e092e01f66313e12069ac8873f2ff1c6c3b22d7f74c870f842957ff85f`;
- derived offline lock SHA-256 `d07866e8a6961193a1285d1514560c46610413f9a05fc19738c286320a98c481`;
- explicit language/vision attention backend: `FLASH_ATTN`;
- FlashInfer sampler and blockscale FP8 GEMM disabled;
- tool parser: `hermes`;
- reasoning parser: `qwen3`;
- served-model alias: `Qwen/Qwen3.8-27B-FP8`;
- exact request SHA-256: `ffa037c93ef061d545eae5a780b194f1a76dc0fd24713783d82c01cfbde469f4`;
- exact image SHA-256: `158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d`;
- one native `take_action({"action":"LEFT"})` request, no retries or redirects;
- exact raw request/response preservation;
- strict reasoning/tool parser validation;
- clean process-group shutdown and GPU-memory release;
- `NvidiaRtxPro6000` in metadata and CLI;
- network disabled;
- `session_timeout_seconds=1800` plus a 1,200-second notebook hard watchdog.

## Exposure ceiling

- one F0-R3 kernel attempt;
- one generation request;
- zero automatic retries;
- at most `0.50h` GPU exposure by Kaggle session timeout;
- at most 1,200 seconds after first notebook cell by the internal watchdog.

## Classification

- another pre-execution rejection: preserve and classify; do not retry;
- hardware mismatch: allocator failure only;
- identity/runtime/backend/startup failure: reject F0 route at that boundary;
- HTTP/parser/native-tool failure: reject F0 contract;
- teardown or memory-release failure: reject lifecycle qualification;
- all gates pass: qualify F0 as a deployment fallback route only.

No F0 outcome changes Qwen behavioral authority, creates a QRM candidate, or authorizes production deployment.

## Launch boundary

R3 is prepared but **not authorized to launch automatically**. After fresh explicit authorization, the only allowed command is:

```bash
uvx kaggle kernels push \
  -p deployment/kaggle/f0-r3-notebook \
  --timeout 1800 \
  --accelerator NvidiaRtxPro6000
```
