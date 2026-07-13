# Model selection: hosted development, local submission

## Decision

Use Alibaba Model Studio's `qwen3.6-27b` endpoint for hosted development, and
target the official `Qwen/Qwen3.6-27B-FP8` checkpoint for the offline Kaggle
submission.

This is the shortest path from the present baseline to a competition-feasible
model:

- ARC Prize reports that the milestone-1 winning Duck harness ran Qwen3.6 27B
  FP8 locally, and our baseline deliberately tests the same LLM-plus-Python
  controller shape. [ARC Prize milestone-1 recap]
- Qwen's model card directly pairs the hosted Model Studio ID `qwen3.6-27b`
  with self-hosted `Qwen/Qwen3.6-27B-FP8` in its agent configuration. It
  documents image input, tool use, thinking mode, and preserved thinking
  history. [Qwen model card: agentic usage]
- The official FP8 release is a 30.9 GB, Apache-2.0 checkpoint. It is a 27B
  causal language model with a vision encoder and a native 262,144-token
  context window. [Qwen FP8 files] [Qwen FP8 model card]
- Qwen documents OpenAI-compatible vLLM and SGLang serving for that exact FP8
  repository, including the `qwen3_coder` tool-call parser. [Qwen serving guide]

The API endpoint is available in Alibaba's Singapore region with a 256K-token
request limit. As of 2026-07-13, the listed international price is $0.60 per
million input tokens and $3.60 per million output tokens in either thinking or
non-thinking mode, with a one-million-token introductory quota. [Model Studio
pricing]

ARC-AGI-3 evaluation has no internet access, so the endpoint is only a
development transport. The final notebook must attach and serve the weights
locally. [ARC-AGI-3 submission requirements]

## What is verified and what is not

Verified:

- The official hosted model ID and the downloadable checkpoint belong to the
  same Qwen3.6-27B release family.
- Both accept images and support agent/tool workflows through OpenAI-compatible
  APIs.
- The FP8 checkpoint is downloadable, openly licensed, and has already been
  used by a winning ARC-AGI-3 Kaggle submission.

Not verified:

- Alibaba does not state that `qwen3.6-27b` serves the FP8 artifact bit for bit.
  Backend precision, kernels, prompt templates, and decoding may differ.
- The current public ARC Prize page does not specify the exact accelerator or
  memory allocation. The winning submission is evidence of feasibility under
  milestone-1 conditions, not a guarantee that every context/batch setting
  fits future evaluation hardware.

Therefore API results are a model-family proxy, not numerical parity evidence.
Before submission, freeze the local checkpoint revision and run paired
API-versus-local traces on the same observations, prompt, tools, seeds, and
sampling configuration. A material policy delta should be treated as a model
change, not dismissed as serving noise.

## Initial run contract

Start in thinking mode with Qwen's documented general-task defaults:
`temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=0`, and
`preserve_thinking=true`. [Qwen best practices]

Pin and log:

- provider region and endpoint;
- model ID `qwen3.6-27b`;
- all sampling and thinking parameters;
- prompt and tool-schema hashes;
- context eviction policy;
- provider response metadata and token counts.

Do not choose the final competition model merely from the milestone result.
Keep Qwen3.6-27B fixed while testing the controller; replace it only if a
same-budget, same-interface model comparison produces repeatable evidence that
outweighs the cost of API/local divergence.

[ARC Prize milestone-1 recap]: https://arcprize.org/blog/arc-prize-2026-milestone-1
[Qwen model card: agentic usage]: https://huggingface.co/Qwen/Qwen3.6-27B-FP8#agentic-usage
[Qwen FP8 files]: https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/main
[Qwen FP8 model card]: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
[Qwen serving guide]: https://huggingface.co/Qwen/Qwen3.6-27B-FP8#serving-qwen36
[Model Studio pricing]: https://www.alibabacloud.com/help/en/model-studio/model-pricing
[ARC-AGI-3 submission requirements]: https://arcprize.org/competitions/2026/arc-agi-3
[Qwen best practices]: https://huggingface.co/Qwen/Qwen3.6-27B-FP8#best-practices
