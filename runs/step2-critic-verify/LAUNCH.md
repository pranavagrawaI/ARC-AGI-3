# Step 2 verification — Duck + critic (question mode) on Kaggle

Launched 2026-09-18 ~18:45 UTC. Kernel `pranavagrawai/taaf-duck-critic-verify-step2` version 1.
https://www.kaggle.com/code/pranavagrawai/taaf-duck-critic-verify-step2

## Question
Does tree `0fc7403` (Duck upstream 7652836 + critic adapter, `DUCK_CRITIC_ENABLED=1`,
`DUCK_CRITIC_MODE=question`) run end to end on Kaggle: setup env carries the flag, the
critic authenticates against the local vLLM, `level_start` fires and parses, the game
loop survives every firing? Secondary, first chance ever: a live `stall_no_change` or
`stall_no_progress` firing (ls20 is the game Duck stalls on; 1200 s cap vs Step 1's 600 s).

## Kill rule
Any firing that ends the game, or a critic error that is not swallowed → the wiring is
wrong; no paired wave until fixed. Zero firings on ls20 with ≥3 consecutive no-change
actions in the events → the trigger is not seeing Duck's batched actions live.

## Parameters (from benchmark_initial.pkl / deploy_target.pkl, verified before upload)
| knob | value |
|---|---|
| games | sb26-7fbdac44, ls20-9607627b |
| passes | 1 (2 episodes, 1 wave, concurrency 10) |
| per-game cap | 1200 s (`max_runtime_s_per_game`) |
| experiment cap | 3600 s |
| critic | `DUCK_CRITIC_ENABLED=1`, `DUCK_CRITIC_MODE=question` (all other DUCK_CRITIC_* default: stall 3/25, 160 tok, 30 s) |
| critic endpoint | actor's vLLM (`vrfai/Qwen3.6-27B-FP8`), `reasoning` flag not sent to vllm provider; `enable_thinking=false` via chat_template_kwargs |
| everything else | identical to Step 1 (`runs/step1-duck-alone/LAUNCH.md` in the duck-harness worktree) |
| source | tree `0fc7403`; `git_status.txt` says DIRTY only because this run dir is untracked |

## Expectations
sb26: L1 in ~70 s (Step 1: 4/4); one `level_start` firing at action 0 (~3 s), possibly a
second at level 2. ls20: no level at 600 s in Step 1; at 1200 s expect ≥25 actions and
therefore at least one `stall_no_progress` firing, plus `stall_no_change` if Duck repeats.
Critic cost per firing ≈ 1/15 of a Duck turn; not expected to move the level count.

## Launch notes
- Dry-run bundle built first (same command with `--kaggle-dry-run`), inspected: env carried,
  setup script compiles, critic package byte-identical to the worktree.
- `--kaggle-duck-public-harness` refuses `--game`; Step 1 ran without it too.
