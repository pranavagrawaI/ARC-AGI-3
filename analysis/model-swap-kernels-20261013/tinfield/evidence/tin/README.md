---
base_model: badtheorylabs/Tinfield-1
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- nvfp4
- modelopt
- experimental
---
# Tinfield-1 NVFP4

Public research conversion of [badtheorylabs/Tinfield-1](https://huggingface.co/badtheorylabs/Tinfield-1), derived from Qwen/Qwen3.8-Flash-Next. The source checkpoint was pinned to `f7c714a97b71342db9e2fc04009d2704c656c80c`.

## Checkpoint format

This Hugging Face repository contains the complete original export, including all 10 safetensors shards:

- Routed experts: NVIDIA ModelOpt NVFP4 W4A4, block size 16.
- PLE embeddings, vision, MTP and other non-expert weights: retained in BF16.
- Architecture: `Qwen4ExpForConditionalGeneration` / `qwen4_exp`.
- The original configuration, tokenizer, processors and tensor index accompany the shards.

The `ple-int8-e579` Kaggle variant is a separate quantized PLE derivative. This Hugging Face checkpoint contains **BF16 PLE**. Load all shards and the configuration from this repository together.

## Conversion and runtime

Conversion used NVIDIA Model-Optimizer recipe commit `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`, ModelOpt 0.46.0, Transformers 5.17.0 and PyTorch 2.13.0+cu130. This export has no additional SFT or RL training.

Use a compatible Qwen4Exp/NVFP4 serving runtime. The recorded functional configuration used vLLM 0.30.0 on one NVIDIA B200, BF16 PLE offloaded to CPU, and approximately 377 GB host RAM. This is a measured configuration; lower-memory hardware requires separate validation. ModelOpt reported that Transformers 5.17.0 was outside its tested version range.

## Evaluation record

The historical 48-episode first-crack probe recorded 20 cracks (41.67%), versus 26/72 (36.11%) for its historical stock baseline. The observed +5.56 percentage points did not meet the preregistered +10-point gate: **NO_GO**. Serving runtimes differed, so this comparison does not isolate a causal model improvement. See `evaluation-summary.json` for the aggregate result and its limits.

These results apply to the original BF16 PLE export. Scores from later Kaggle runs with INT8 PLE apply to that separate checkpoint/runtime/harness combination.

## Release contents and integrity

The 2026-10-05 publication cleanup retains all model shards and serving assets byte-for-byte. `release-manifest.json` records their sizes and hashes. Internal calibration examples, per-game probe records, machine diagnostics and conversion scripts have been archived outside this public release. The model card retains the aggregate historical evaluation result.

## License and credits

[Qwen Community License 1.0](LICENSE). Credits to badtheorylabs for Tinfield-1, Qwen for the base model, and NVIDIA Model Optimizer for the quantization tooling.
