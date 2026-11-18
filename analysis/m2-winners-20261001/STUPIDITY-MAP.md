# Stupidity map: where dfranzen's system (our new incumbent) and our old one waste the binding budget (2026-10-01)

Method: a time and waste study of dfranzen's actual run (537 logged requests, SGLang server log, transcripts), the scorer source,
and our 8 full public-25 runs. Scripts in this directory: `wastewalk.py`, `transcript_stats.py`, `context_inventory.py`,
`action_slack.py`, `action_matched.py`. The forum reader is `forum/forum.py` (kagglesdk discussions + search API).

## 0. The rules, read from source (arc_agi 0.9.9 `scorecard.py`, `api.py`; arcengine `base_game.py`)
- Level score = min(115, (human_baseline / actions)^2 x 100). Every ACTION1-7 counts, and so does every RESET (`inc_reset_count` adds an action).
  Deaths count too: all actions at a level accumulate until it is completed.
- Game score = sum(level_index x level_score) / sum(level_index), capped at the completed-weight fraction. Later levels weigh more.
- Competition mode is one play per game: `_get_or_create_environment` refuses a second environment for a game in the card, and a
  RESET at action_count 0 is intercepted. (`EnvironmentScoreList.score` takes the max over runs, but competition mode allows one run.)
  After a WIN, RESET does a full reset, which would start a new play. That only matters for already-won games.
- Human baselines on public-25: median 60 actions per level (L1 median 30), p10 22. 92% of levels allow >= 20 actions.
  Credit stays full up to ~0.93x baseline (the cap is 115), so roughly the first 20 actions of almost any level cost no score.

## 1. Throughput decomposition: our 10x gap is mostly basics we got wrong
Same 10 public games, levels per million generated tokens:

| run | tokens | levels | L/Mtok | aggregate tok/s |
|---|---|---|---|---|
| dfranzen (25 min, 10-way) | 0.89M | 40 | 45.2 | ~750 (saturated) |
| our KF (132 min, 28-way) | 0.81M | 27 | 33.2 | ~250 |
| our P2 | 1.13M | 37 | 32.8 | ~380 |
| our P1 | 1.17M | 31 | 26.5 | ~380 |

So the gap is about 1.4x per token (harness) times about 2-3x tokens per hour (serving), about 3-4x per wall-clock hour. The hidden-score gap is 9.4x
(27.89 vs KF 2.96). The rest is consistent with our harness collapsing on harder (hidden) games: our public->hidden drop is 3.4x,
dfranzen's is about 1x (estimated). Candidate causes, all retired by adopting dfranzen's stack: 32K context, and the
history-wipe when a turn exceeds 30 exchanges (codex probe). **Test:** P's pending hidden score, and dfranzen's notebook run in our account.

## 2. dfranzen's own system: what is NOT waste (don't spend effort here)
- GPU idle: none. Median running is 9-10 of 10, the queue is ~0, and **throughput saturates at ~750 tok/s** (8->10 running adds only 5%).
- Prefill: 94% of prompt tokens are cached.
- Startup: server READY at 531 s = 1.6% of 9 h.
- Deaths: 5 in the whole run. Tool timeouts: 2. Truncated output to max_tokens: 1.
- History wipes: none. The 3 collapses are the designed 58K block drains.

## 3. What IS waste, ranked by size
| # | waste (Ohno category) | evidence | lever | cost |
|---|---|---|---|---|
| S1 | **Mis-specified objective: hoarding free actions, spending scarce tokens** (over-processing) | Prompt: "Optimize for as few in-game actions as possible". 31/40 solved levels under 0.93x baseline; 410 free actions unused vs 946 used. **25% of all tokens spent before a level's first action** (median 4K per new level, up to 20.7K). 86% of generation is thinking; 53% of requests only inspect. r11l L3: 1 action of 51 free, 37K tokens. | Tell the model the true economics: actions free up to ~human count (~20/level safe), credit falls quadratically beyond it, tokens/time are the scarce budget, later levels weigh more. Encourage discriminating probes over deliberation. Prior art: gedouluhui (+28% hidden from a true-scoring + time-budget prompt); lordhansolo ca1bd02 "act without full certainty". | prompt only |
| S2 | **Duplicate inventory in context** | Late-game context: 51% own past thinking, **~19% repeated per-turn boilerplate** (82% of user text repeats verbatim), 10% tool-result evidence, 9% system. | Keep boilerplate only in the newest user message; strip it from history. More real evidence per 116K window, fewer drains. | small harness change |
| S3 | **Throughput ceiling** | SGLang saturates at ~750 tok/s at 10 streams (per-seq 95 -> 76 tok/s); MTP accept 2.67/4. lordhansolo's vLLM e975732 logs 976 tok/s at 14 running. | Tune speculative steps (3 vs 2) at saturation; or adopt the lordhansolo runtime under the dfranzen harness. | GPU bench |
| S4 | Motion waste: tool errors | 41/531 calls (8%) error: NameError on harness names (`last_action_timeline` vs `last_animation_timeline`), a sandbox-blocked `open()` killing a 4K-token search turn. | Alias common misnames; make blocked builtins no-op with a warning instead of killing the cell. | small |

## 4. What the forum says about the teams above dfranzen (Tufa 50.65, Yi-Chia Chen 40.80)
- Neither published. Tufa M1 writeup: "expect this approach to scale with better small models and additional post-training." Yi-Chia Chen
  (threerabbits) does agentic RL and on-policy distillation (AIMO). Training is plausible for both. But blakewest (SFT on H200s, Qwen Max/Astra
  teachers), AAAAAtjc (RL pipeline on an H200 node) and Scott Le Grand (+2 public, 0 hidden) all report **no gain** from fine-tuning.
- Serguei Makarov (743723), controlled study: per-game outcomes are bimodal. In 83% of stuck-vs-success forks, the stuck run had already
  tried the decisive action type; the failure is **early goal judgement, then defended**. He also: "go back to basics... results striking."
- The late-September group jump (20 -> 35 -> 45) coincides with the long-context fp8-KV serving wave (Pennyroyal 2.5.x, vLLM main).

## 5. Next decisions (not launched; GPU quota 7.06 h until Sat 00:00Z)
1. Run dfranzen's notebook unchanged in our account on full public-25 (control; it also qualifies the stack). Optionally submit it once to measure our
   account's hidden baseline.
2. One-change arms against that control, read by mechanism counters first (tokens before first action, actions/baseline, L/Mtok), score second:
   S1 economics prompt; S2 history hygiene. Noise is large (sd ~6 levels per single pass), so use counters plus repeat runs.

## 6. Control run in our account (2026-10-01)
- `pranavagrawai/dfranzen-m2-raw` v1 (byte-identical): 10 demo games, 25 min, **44 levels, mean 41.75** (dfranzen 40 / 36.56). Submitted as **56747536**.
- `pranavagrawai/dfranzen-m2-control` v1 (all 25 public games, 7,254 s/game, concurrency 25, 10 streams): **107 levels, 9 games won, mean 52.48**
  (our P: 51-58 levels, 11.75-16.11). 4.56M tokens, 87% thinking, 94% cached, ~745 tok/s saturated for 2 h, 8% tool errors (ImportError on blocked `time`/`builtins`/`inspect`).
- Waste in the control: 6 games ended at L0-L1 and consumed ~0.94M tokens (21%) for <3 points. Pre-first-action deliberation is 16%. 90/107 solved
  levels came in under 0.93x baseline (median 0.59x). Stuck levels split: far under baseline (s5i5 L2 16/89 with 194K tokens, su15 L4) or far over (ka59 3.7x, sc25 2.0x).
- S1 arm `pranavagrawai/dfranzen-m2-s1` v1 pushed 08:43Z (control + economics prompt only).

## 7. Six questions, answered from the control run (2,341 requests, 4,921 actions; `six_questions.py` + inline probes)
1. Repeated work: 34% of code lines are re-typed from earlier calls in the same game. 45% of calls re-parse the board from ascii/segmentation
   (`a=current_frame.ascii.split('\n')` x402, `seg=current_frame.segmentation` x180, `from collections import ...` x159). Shape constants are re-typed x35 in one game.
   ~19% of late context is repeated per-turn boilerplate.
2. Information lost: 89 history drains (3.6/game) drop ~59K tokens each, with no carried summary (WM is off). The next 3 requests cost +19% tokens and act 38% vs 45%.
   The sandbox wipes variables every call: 26/35 NameErrors are names defined earlier in the game. Retained functions keep code, not data.
3. Mechanically eliminable uncertainty: harness API shape (HistoryEntryView has no .step; _LazyAnimation has no .items; 'list' has no .split),
   which imports are allowed (27 ImportErrors), what changed (113 inspect-only diff calls + ~125 animation reads), what each action does
   (learned by thinking; 72K coordinate tuples written in thinking). 56% of calls take no action.
4. Known-useless actions: small. 6% of actions leave the interior unchanged; only 1% repeat a (state, action) already known to be a no-op; 0 repeated fatal moves;
   13% return to an interior state already seen this level; 8% re-execute an already-known transition.
5. What code could know: board parsing (45% of calls), object positions/diffs, click coordinates (MOUSE calls hand-typed with literal row/col x200+),
   grid transcription into thinking (0.5% of chars, but 72K coordinate tuples), counting/indexing ("let me recount/index carefully" x1,137).
6. Capability compensating for infrastructure: 182 tool errors (8%) — sandbox resets (NameError), blocked stdlib (time/builtins/inspect/pickle), API mismatches,
   StaleState guard trips (25); raw grid deliberately hidden (forces ascii re-parsing); the per-call namespace reset forces re-derivation; the model reads ~19% duplicated boilerplate.

## 8. S1 result (dfranzen-m2-s1 v1, 2026-10-01 09:04-11:07Z)
Mechanism live (S1_ACTIVE; new prompt in dumps). Behaviour moved as intended: solved-level median actions/baseline 0.59 -> 0.65, levels over baseline 14 -> 19,
stuck-level actions/baseline 0.35 -> 0.71. Outcome: **98 levels / 39.52 vs control 107 / 52.48**. Per game: 8 better, 9 worse, 8 same; s5i5 +5, tr87 -5, cn04 -4.
The level delta (-9) is inside single-pass noise (sd ~6+); the score drop is larger because losses landed on high-weight levels and more levels went over baseline.
Verdict: **not retained**. More probing produced more actions, not more understanding, which is consistent with Makarov's judgement-not-coverage finding.

## 9. Replicate + hidden baseline (2026-10-02)
- Raw dfranzen submission 56747536: **23.87 hidden** (dfranzen's own 27.89; our P 3.08). The incumbent reproduces on hidden within noise.
- control-r2 (identical code, 100 min/game): 100 levels, 8 won. At the 100-min cut (`levels_at_time.py`): control 103, replicate 100, S1 95.
- Noise: identical runs differ on 13/25 games (cd82 6 vs 0, tu93 8 vs 3, s5i5 1 vs 6). Per-game diff sd 2.2 -> **sd of a 25-game total diff is ~11 levels**.
  S1 sits 6.5 levels below the control mean: **inconclusive, not shown harmful**. A single-pass arm needs a gain of ~20+ levels to count, or several runs per arm.
