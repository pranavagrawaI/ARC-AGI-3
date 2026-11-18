# Harness input-serialization failure; target hardware confirmed

The competition-bound run received NVIDIA RTX PRO 6000 Blackwell Server Edition,
SM120 (`hardware.json`). The immutable model verification returned `matched`.
The run then failed before the arm loop or any SGLang server launch:

```text
TypeError: Object of type BatchEncoding is not JSON serializable
```

The traceback points to runner.py line 195, saving frozen-input.json. At lines
192-194, the runner calls apply_chat_template without return_dict and assigns
the result directly to input_ids. The installed package inventory records
Transformers 5.12.1. Its tagged source defaults return_dict=True and returns a
BatchEncoding; return_dict=False selects out["input_ids"] instead.

Source: https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/tokenization_utils_base.py#L2714

This is my harness API-assumption error, not an FP8, MTP, manifest or allocation
failure. The local verifier tested notebook identity, push construction and
stream metric arithmetic but bypassed real tokenizer input preparation.

Narrow proposed correction: pass return_dict=False explicitly, then require a
nonempty flat list of integer token IDs and JSON-roundtrip the full request.
Do not stringify BatchEncoding with a generic JSON fallback: the native engine
endpoint needs actual input IDs. Exercise this preparation path with the pinned
tokenizer locally before spending another allocation.

No runtime code was changed and no retry was launched during this diagnosis.
No arm produced a benchmark result. Raw evidence is retained under kaggle-output,
especially isolated-engine-20260905/runner.log, hardware.json,
model-verification.json and qwen38-fp8-r8l3b-bootstrap/runtime-packages.json.
