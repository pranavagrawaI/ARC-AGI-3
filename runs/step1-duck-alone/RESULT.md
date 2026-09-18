# Step 1 result — Duck alone on Kaggle: gate PASSED

Kernel `pranavagrawai/taaf-duck-alone-step1` v1. QUEUED 09:10Z → RUNNING 09:12Z → COMPLETE 09:33Z.
Output in `kaggle-output/` (events.jsonl, viewer data, transcripts, benchmark.json, vLLM log).

## Substrate
- GPU allocated: **NVIDIA RTX PRO 6000 Blackwell Server Edition** ×1 (setup-script check passed).
- vLLM wheelhouse install + model load + warm-up: 9.6 s → 571.6 s (**~9.4 min**).
- Games: 571.6 s → 1171.9 s. All 10 episodes ended `gave_up` at the 600 s cap (no crashes, no
  parser/sandbox failures in the kernel log).

## Outcome (10 episodes, 1 wave, concurrency 10)
| episode | levels | actions | turns | s/turn | s→L1 | actions→L1 | board_changed % |
|---|---|---|---|---|---|---|---|
| sb26 p0 | **1** | 24 | 10 | 46.9 | 133 | 9 | 100 |
| sb26 p1 | **1** | 17 | 14 | 41.1 | 291 | 13 | 88 |
| su15 p0 | **1** | 31 | 15 | 37.1 | 196 | 19 | 94 |
| su15 p1 | **1** | 23 | 9 | 60.3 | 322 | 23 | 96 |
| lf52 p0 | 0 | 11 | 13 | 45.8 | – | – | 100 |
| lf52 p1 | 0 | 12 | 15 | 39.9 | – | – | 100 |
| ft09 p0 | 0 | 43 | 8 | 70.9 | – | – | 77 |
| ft09 p1 | 0 | 8 | 11 | 50.2 | – | – | 0 |
| ls20 p0 | 0 | 13 | 11 | 54.1 | – | – | 100 |
| ls20 p1 | 0 | 19 | 12 | 45.6 | – | – | 84 |

TAAF mean score 0.98. `s/turn` = span between first and last transcript header ÷ turns (undercounts the last turn).

## Kill rule
"No L1 on sb26/su15 → budget problem." L1 reached 4/4 on those games. **Proceed.**

## What this measures that nothing did before
- **Duck turn latency on Kaggle at 10-way concurrency: median ~46 s/turn.** Faster per turn than
  Duck's own B200 example run (~86 s/turn at 16–32-way), but a 10-turn game is still ~8 min.
- Time to L1 on sb26/su15 is 2–4× the example-run medians (133–322 s vs 66–78 s); still inside 600 s.
- lf52/ft09/ls20 (example-run medians 341–707 s to L1 on B200) are out of reach at 600 s: 0/6.
- Nobody reached L2 in 600 s; L1 on sb26/su15 is saturated (4/4). So at this cap the paired test
  would have only time-to-L1 on two games as signal.

## Implication for Step 3 (recommendation, not decided)
Raise the per-game cap to 1200 s. Cost model: one wave of 10 episodes = cap + ~10 min startup;
2 arms × 5 games × 4 passes = 40 episodes = 4 waves at concurrency 10 → ~90 min kernel at 1200 s.
Keep concurrency fixed at 10 across arms (it sets per-turn latency). A critic firing must cost
well under one Duck turn (~46 s) or each firing buys back less than it spends.
