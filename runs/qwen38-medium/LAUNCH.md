# qwen38-medium launch — 2026-09-22 12:52Z

Kernel `pranavagrawai/taaf-duck-qwen38-medium` v1, pushed 12:52:16Z from tree `58e0cc0`.
https://www.kaggle.com/code/pranavagrawai/taaf-duck-qwen38-medium

Pushed **in parallel** with `taaf-duck-qwen38-qualify` (queued since 11:05Z, still QUEUED at
push time). The card said to serialize; that was tidiness, not correctness — Step 3 and
periodic-v1 both ran two GPU kernels simultaneously, and the queue, not the GPU, is the
bottleneck today. Each arm is ~0.4 h of quota (22.95 h remaining).

Dry-run verified before push:

| check | value |
|---|---|
| `CHAT_TEMPLATE_KWARGS` in bundle | `{"preserve_thinking": true, "reasoning_effort": "medium"}` |
| served model | `Qwen/Qwen3.8-27B-FP8` from `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` |
| `DUCK_CRITIC_*` keys | 0 |
| recipe | identical to arm 1: 5 games x 2 passes, 600 s, 10-way, 75 min envelope |

Only difference from arm 1: the reasoning_effort kwarg. With `medium` the rendered system
prompt is byte-identical to the Qwen3.6 render, so this arm isolates the checkpoint.

Readback for both: `python3 runs/qwen38-qualify/readback.py <kaggle-output>`.
