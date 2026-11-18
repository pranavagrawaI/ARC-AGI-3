# 20260921-v12-sb26-critic-off result

Kernel `pranavagrawai/tycho-nooa-v12-sb26-critic-off` v1. COMPLETE 2026-09-21T21:48Z.
GPU: NVIDIA RTX PRO 6000 Blackwell, 97887 MiB. `result.json` status `passed`.
Raw: `analysis/deployment-probes/20260921-v12-sb26-critic-off/kernel-output/`.

## Card classification: interpret

Assessor stamp is `INVALID` only because `game-stops.jsonl` lives on the parent
workspace, not in each GAME_OVER-split session dir. Parent file has two
`wall_budget` stops. Critic-off, contingent-off, eviction, sequential walls, and
GPU all match the card. Do not treat that path mismatch as a 0-level ceiling.

## Factor check

All three sessions: `epistemic_critic=0`, `contingent_arm=off`.
`model_calls.jsonl` is 100% `actor` (40+148+155). Zero `request_purpose=critic`.
Eviction fired. No `EVICTION_ERROR`. Two sequential `# episode` markers, one
game on `/api/games` each time.

## Time to L1 vs contrast (critic-on sb26 pass 1: 28 min episode / 11 min winning life / 85 actions)

| Episode | Stop | Env actions | L1 | Episode clock | Winning life |
|---|---|---|---|---|---|
| sb26 pass 0 | wall 1941 s | 45 | **yes** SPACEBAR at count 44 | **32 min** (20:44:30 → 21:16:51) | same (no RESET) |
| sb26 pass 1 | wall 1894 s | 97 | **yes** SPACEBAR at count 80 | **17.5 min** (21:16:53 → 21:34:22) | **12.5 min** after RESET at 21:21:47 |

Pass 2 repeated L1. Episode-clock latency did not drop below the 28/11 contrast
in a way that survives variance: this kernel alone is 32 min vs 17.5 min.
Winning-life 12.5 min vs 11 min is noise. Duck 133–322 s is still missed.

## Reading

Critic/contingent off is established. L1 on sb26 at 1800 s is repeatable without
the critic. The critic is **not** shown as the tempo lever. Do not keep a
critic-tax story. Next licensed card is one Duck cap (batching or letter
boards), not four, if the goal is sb26 L1 ≤ 322 s.
