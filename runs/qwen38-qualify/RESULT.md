# qwen38-qualify result — PASS, with the xhigh confound measured

Kernel `pranavagrawai/taaf-duck-qwen38-qualify` v1. QUEUED 11:05Z (3 h 35 m) -> RUNNING 14:40Z ->
COMPLETE 14:58Z. Output in `live/kaggle-output/`. Readback: `runs/qwen38-qualify/readback.py`.

## Kill rule (CARD.md): PASSED
| check | result |
|---|---|
| vLLM 0.19.0 serves the 3.8 checkpoint | yes — ready at **380 s**, vs 571 s for 3.6 (loads faster) |
| tool calls parse | **124/124** `finish_reason=tool_calls`, 0 recovered from markup |
| L1 on sb26/su15 | **3/4** (fail was 0/4; 3.6 baseline 4/4) |

The chat-template prediction held exactly: markup is identical between checkpoints, so
`qwen3_coder` parses 3.8 without a single fallback.

## Measured against 3.6 on the identical recipe (10 episodes, 600 s, 10-way)
| | 3.6 | 3.8 @ xhigh | |
|---|---|---|---|
| per-stream decode | 18.7 tok/s | **26.3 tok/s** | **+41 % faster** |
| output tokens / call (est.) | 514 | **941** | +83 % |
| reasoning share of output | 59 % | **80 %** | |
| calls / turn | 1.67 | 1.44 | |
| median s/turn | 46.3 | **55.2** | **19 % slower** |
| actions executed (10 eps) | 201 | **120** | −40 % |
| levels (10 eps) | 4 | 3 | |
| turns per episode (min / median) | 8 / 12 | **1** / 10 | |

**The checkpoint is faster; the arm is slower.** 3.8 decodes each token 41 % quicker, then spends
83 % more tokens per call, and the extra tokens are reasoning (80 % of output). The decode win is
spent and overdrawn.

Pathological case: `ls20_p0` ran **1 turn and 0 actions** — its first call never returned and was
cut by the 600 s cap. (A single trailing timeout per episode is normal and appears in 8/10 of the
3.6 episodes too: it is the cap cutting the last call. What is new here is a call long enough to
swallow an entire game.)

## What this does and does not establish
Establishes: 3.8 serves on Duck's stack, parses, and plays. Its raw decode is materially faster.
Does not establish: that 3.8 is better or worse at ARC-AGI-3 — this arm carries the template's
`xhigh` reasoning instruction, so checkpoint and reasoning effort are confounded by construction.

## Next: arm 2 (`runs/qwen38-medium`)
`reasoning_effort=medium` renders byte-identically to 3.6's prompt. If it restores ~514 tok/call
at 26.3 tok/s per stream, a call costs ~19.5 s against 3.6's ~27.5 s — a materially faster turn.
That is the arm that decides whether 3.8 belongs in a commit run. Projection only; the run decides.
