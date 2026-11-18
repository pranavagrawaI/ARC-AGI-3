---
license: other
license_name: qwen
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: finetune
pipeline_tag: text-generation
tags:
- qwen
- reasoning
- efficient-reasoning
- bf16
---

# Qwen3.8-Flash-Next-AZ-Focus (BF16)

**Qwen3.8-Flash-Next with the second-guessing turned down: about 40% less thinking at Qwen's default effort, with
agentic coding unchanged.**

Qwen3.8-Flash-Next thinks carefully, but on hard questions a lot of that thinking is hesitation: *"Wait…"*,
*"Hmm, maybe…"*, re-checking answers it already had. We optimised the model's weights against that habit, while leaving
alone the checking it does as an agent (running tests, inspecting output before it submits). AZ-Focus still reasons step
by step. It just loops less.

This is the full-precision release: the same files, layout and architecture as
[Qwen's release](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), with some output-projection weights changed. Use it
anywhere Qwen3.8-Flash-Next BF16 runs, or as the source for your own quantizations.

Made by [AxonZeta](https://axonzeta.ai), an independent AI lab in Manchester, UK.

## Results

Measured on our EXL3 4.05 bpw conversion of these weights
([Qwen3.8-Flash-Next-AZ-Focus-EXL3](https://huggingface.co/AxonZeta/Qwen3.8-Flash-Next-AZ-Focus-EXL3)) against a stock
conversion of Qwen's release with identical settings, on the same server and hardware, at the default effort (xhigh):

| Benchmark | Original | AZ-Focus | Thinking tokens |
|---|---|---|---|
| MMLU-Pro (48) | 83.3% · 6,948 | 81.2% · 4,739 | −32% |
| BIG-Bench Hard (48) | 87.5% · 2,346 | 83.3% · 618 | −74% |
| GPQA Diamond (20) | 90.0% · 13,622 | 90.0% · 7,372 | −46% |
| AIME 2025 (12) | 100% · 36,305 | 100% · 24,833 | −32% |
| **All 128 answers** | **112 correct** | **109 correct** | **−39%** |
| SWE-bench Verified, first 100 (mini-SWE-agent) | 96 | 98 | |

Thinking on, temperature 1.0, top-p 0.95, top-k 20, 128k output limit; no answer hit the limit. A 128-answer set
(64 questions, two seeds), so a few answers either way is within noise. We haven't benchmarked the BF16 weights
directly; we checked that the change survives quantization to EXL3. None of these questions was used to build the model.

## Effort setting and limits

- **Use it at the default (xhigh) effort.** That is where the hesitation is. Lower efforts haven't been measured on this
  build.
- **Set an output limit** (`max_tokens`): on rare hard problems the model can keep exploring for a very long time.
- Other languages, very long contexts, vision and multi-hour agent sessions haven't been measured separately yet.

## License

[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) (see `LICENSE`), as for
the original model. Note its terms for very large products and for Model-as-a-Service / AI work-assistant businesses.

## Credits

- **Qwen** for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

Questions, or want to build on this with us? hello@axonzeta.ai
