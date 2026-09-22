# qwen38-qualify launch — 2026-09-22 11:05Z

Kernel `pranavagrawai/taaf-duck-qwen38-qualify` v1, pushed 11:05:26Z from tree `366558e`
(branch `qwen38-submission`, worktree `~/projects/arc-agi-3-reference/duck-qwen38`).
https://www.kaggle.com/code/pranavagrawai/taaf-duck-qwen38-qualify

Dry-run bundle verified first (`runs/qwen38-qualify/dryrun`), then pushed unchanged:

| check | value |
|---|---|
| dataset sources | source + `driessmit1/arc3-vllm-h100-wheelhouse-v3` + `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` |
| served model | `Qwen/Qwen3.8-27B-FP8` (setup env + solver both) |
| 3.6 references in bundle | none |
| `DUCK_CRITIC_*` keys | 0 |
| pickled per-game cap | 600 s; kernel `max_runtime_s` 4500 s |
| games / passes | 5 / 2 = 10 episodes, 1 wave, concurrency 10 |
| accelerator / internet | NvidiaRtxPro6000 / off |
| soft-deadline guard | present in the bundled notebook (`Applies in a real submission`) |
| Kaggle token | refreshed, expires 20:08Z |

Expected: ~10 min vLLM startup (3.6 took 9.4 min; 3.8 weights are 28.7 GiB), then 600 s of
games, COMPLETE ~11:27Z. Read against the 3.6 Step 1 table in
`duck-harness/runs/step1-duck-alone/RESULT.md` (46 s/turn, L1 4/4 on sb26/su15).
