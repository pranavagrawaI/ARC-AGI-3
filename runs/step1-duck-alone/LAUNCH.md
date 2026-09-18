# Step 1 — Duck alone on Kaggle (incumbent gate)

Launched 2026-09-18 ~09:20 UTC. Kernel `pranavagrawai/taaf-duck-alone-step1` version 1.
https://www.kaggle.com/code/pranavagrawai/taaf-duck-alone-step1

## Question
Does Duck, exactly as published (upstream `7652836`, its own Qwen3.6-27B-FP8 + wheelhouse
datasets), complete level 1 under our per-game cap on Kaggle's RTX PRO 6000?
Secondary: seconds per Duck turn on this GPU at concurrency 10 (never measured).

## Kill rule
No L1 on sb26/su15 → the 600 s cap cannot discriminate; fix the budget before touching the critic.

## Parameters (from benchmark_initial.pkl / deploy_target.pkl, verified before upload)
| knob | value |
|---|---|
| games | sb26-7fbdac44, su15-1944f8ab, lf52-271a04aa, ft09-0d8bbf25, ls20-9607627b |
| passes | 2 (10 episodes, 1 wave, concurrency 10) |
| per-game cap | 600 s (`max_runtime_s_per_game`) |
| experiment cap | 4500 s → notebook soft end at 65 min |
| analyzer timeout | 900 s per request |
| model | vrfai/Qwen3.6-27B-FP8 via `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` |
| vLLM | `driessmit1/arc3-vllm-h100-wheelhouse-v3`, max-model-len 65536, TP 1 |
| actor context window | 32768 (Duck JSON config, exported into setup env) |
| tool_steps / yield | 0 (unbounded) / 60 s |
| accelerator | NvidiaRtxPro6000, passed as push-time `--accelerator` |
| internet | off; env files from `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files` |
| source | upstream duck-harness `7652836`, no code changes |

## Expectations from example-run (B200, 45-min cap, 16–32 concurrent)
sb26/su15: L1 in 20/20 passes, median 66–78 s. lf52/ft09: 15/20, median 341–701 s.
ls20: 4/20, median 707 s. If Kaggle turn latency is ≥2× B200-under-load, lf52/ft09/ls20 are out of reach.

## Launch notes
- `make kaggle-duck` hung in a `$(shell …)` with an orphaned pipe; launched via the CLI with the
  Makefile's exported env reproduced in `/tmp/duck-step1-env.sh` (values recorded in setup_commands.json).
- Dry-run bundle inspected first (`../step1-duck-alone-dryrun`).
