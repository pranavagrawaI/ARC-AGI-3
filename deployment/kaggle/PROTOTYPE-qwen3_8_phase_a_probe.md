# PROTOTYPE — Qwen3.8 FP8 offline Phase A probe

**Question:** Can the immutable `Qwen/Qwen3.8-27B-FP8` checkpoint at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` and SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` cold-load without network access on Kaggle's RTX PRO 6000 and turn one ARC-like image into one parsed native tool call?

This is a throwaway deployment probe. It does not run the Solver, consume ARC actions, submit to Phase B, measure concurrency, or compare behavior with the hosted model.

## One local action

```bash
python3 PROTOTYPE-qwen3_8_phase_a_probe.py \
  --self-test \
  --output-dir /tmp/qwen38-phase-a-probe-self-test
```

The self-test starts a fake OpenAI-compatible server and exercises image generation, request serialization, evidence capture, reasoning extraction, native-tool parsing, and result classification. Its claim ceiling explicitly excludes the model, runtime, GPU, and Kaggle.

## Real Phase A action

The real run requires two Kaggle inputs:

1. the public Kaggle dataset `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`, whose 80 visible files and sizes match the immutable Hugging Face revision (Kaggle omits only the non-runtime `.gitattributes` transport file); and
2. public Kaggle dataset `mahdiseddigh/kaggle-t4-stack-2026-08-28`, version 1, whose lockfile pins SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd` and whose SGLang wheel and lockfile hashes are embedded in the probe.

Build the notebook after those versioned sources exist:

```bash
python3 scripts/build_phase_a_probe_notebook.py \
  --kernel-id pranavagrawai/prototype-qwen3-8-fp8-phase-a-qualification \
  --model-dataset-source jakobbrggen/qwen3-8-27b-fp8-hf-snapshot \
  --runtime-dataset-source mahdiseddigh/kaggle-t4-stack-2026-08-28

uvx kaggle kernels push \
  --path probe-notebook \
  --timeout 1800 \
  --accelerator NvidiaRtxPro6000
```

The builder also writes `machine_shape: NvidiaRtxPro6000` into `kernel-metadata.json`; the explicit push argument is retained as a fail-closed override because generic GPU metadata produced P100 runs in R5 and R6. Use **Save & Run All**, not **Submit to Competition**. The notebook installs only from the attached wheelhouse with `--no-index`, validates every checkpoint file against `PROTOTYPE-qwen3_8_fp8-manifest.json`, launches the frozen FP8 route without speculative decoding, sends one request, and preserves all evidence under `/kaggle/working/qwen38-fp8-phase-a-probe/`.

## Pass boundary

A pass requires all of the following:

- observed GPU name contains `RTX PRO 6000`;
- all 80 runtime model files match the immutable revision by size and SHA-256; the optional `.gitattributes` transport file may be absent;
- runtime provenance binds Kaggle dataset version 1, SGLang commit `8e04f66a7046fa083cda3f5b8e8325632a5af8fd`, SGLang wheel SHA-256 `9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d`, and lockfile SHA-256 `d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c`;
- SGLang reaches `/v1/models` within 1,200 seconds;
- the response contains non-empty parsed reasoning;
- the response contains exactly one native `take_action` call with `{"action":"LEFT"}`.

The image contains a red square left of a yellow square. The expected action therefore depends on the image rather than a value disclosed in the prompt.

## Evidence

Each run directory contains the exact request and response, generated PNG, model verification, server command and logs, readiness timing, system/GPU snapshots, event log, and `result.json`. Preserve failed directories unchanged; a retry gets a new directory.
