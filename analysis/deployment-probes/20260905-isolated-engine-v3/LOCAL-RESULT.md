# Actual tokenizer test passed; bounded v3 launched

The local test uses Transformers 5.12.1, Tokenizers 0.22.2 and Jinja2 3.1.6,
matching the v2 runtime inventory. The tokenizer JSON, tokenizer config, model
config and chat template were downloaded without model weights and SHA-256
checked against v2's successful immutable manifest verification.

`LOCAL-TEST.json` records:

- The original call returns BatchEncoding and reproduces the exact JSON error.
- The corrected packaged runner explicitly requests return_dict=False and
  validates a nonempty flat integer list.
- All 81 token IDs equal the original BatchEncoding's input_ids field.
- Frozen input saving succeeds, and the actual runner's full request literal
  JSON-roundtrips for warmup, fixed-output and natural-output parameters.
- The tested runner's bytes are present in the delivered notebook.

No local model inference or CUDA execution is claimed. The initial tokenizer-only
environment needed Jinja2 installed before chat-template execution; it was then
pinned to the deployed version. No target model or prompt changes were needed.

Build checks, focused Ruff checks, and the installed Kaggle push-construction
test pass. The latter uses zero transport and confirms ARC-AGI-3 binding,
NvidiaRtxPro6000, private/offline mode and 3600-second timeout.

Launch: `kaggle kernels push --path deployment/kaggle/isolated-engine-20260905-v3 --timeout 3600 --accelerator NvidiaRtxPro6000`.
Separate kernel: `pranavagrawai/qwen38-isolated-engine-20260905-v3`.
The prior failed candidates remain unchanged. GPU measurements are pending.
