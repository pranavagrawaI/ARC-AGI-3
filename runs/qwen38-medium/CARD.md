# qwen38-medium — isolate the checkpoint from the xhigh reasoning instruction

Frozen 2026-09-22, before push. Push only after `taaf-duck-qwen38-qualify` reaches a terminal
status (one GPU kernel at a time keeps the queue honest and the comparison same-day).

## What the first arm actually tests
Rendering both chat templates locally with Duck's exact launch kwargs
(`{"preserve_thinking": true}`, tools, add_generation_prompt) shows the Qwen3.8 template
prepends to every system prompt:

    Reasoning effort is set to xhigh. Please think carefully through the task, validate key
    assumptions, consider plausible alternatives, and prioritize correctness, consistency,
    and clarity in the final answer.

| reasoning_effort | rendered chars | injected line |
|---|---|---|
| unset (= arm 1) | 1410 | xhigh |
| `medium` | 1201 | none — **byte-identical to the 3.6 render** |
| `low` | 1339 | brevity |
| `high` | — | template raises `Unexpected reasoning effort` |

So arm 1 = checkpoint + xhigh instruction. This arm = checkpoint alone.

## Question
With the prompt held byte-identical to 3.6's, does Qwen3.8-27B-FP8 match or beat Qwen3.6 on
the Step 1 recipe — and how much of any arm-1 slowdown was the xhigh instruction?

## Change from arm 1 (one knob)
`DUCK_CHAT_TEMPLATE_KWARGS='{"preserve_thinking": true, "reasoning_effort": "medium"}'` on the
launcher. `inference/framework/kaggle.py` now threads this into vLLM's
`--default-chat-template-kwargs`; unset reproduces the historical literal exactly.
Everything else identical to `runs/qwen38-qualify` (5 games x 2 passes, 600 s, 10-way).

## Reading (three-way, same recipe, n=10 each)
| | 3.6 (Step 1) | 3.8 xhigh (arm 1) | 3.8 medium (this) |
|---|---|---|---|
| levels on sb26/su15 | 4/4 | | |
| median s/turn | 46 | | |
| output tok/call | 514 (est.) | | |
| reasoning share | 59 % | | |

Attribution rule, written in advance:
- arm1 slow **and** this arm ~= 3.6 -> the xhigh instruction is the cost, not the checkpoint.
- both arms slow -> the checkpoint is slower on this workload; 3.6 stays.
- this arm faster or deeper than 3.6 -> 3.8 at medium is a real candidate for the commit run.

n=10 per arm; ±1 level on a game is noise. This screens, it does not settle.
