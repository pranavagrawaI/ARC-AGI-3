# qwen38-fixes — do the step-1 fixes buy levels? (frozen 2026-09-23 before push)

## What changed (tree 7769142 + launcher env), and the evidence for each
| fix | knob | evidence (runs/qwen38-learnings/STEP1.md) |
|---|---|---|
| turn yield 60 s -> 300 s | `LOCAL_ANALYZER_YIELD_SECONDS=300` | 45 % of 3.8 turns ended with no action (3.6: 22 %); one call takes ~142 s median at 28-way, so 60 s ends nearly every turn after its first call; ls20 re-ran the same first survey 42x |
| game-over retry note | `DUCK_PROMPT_GAMEOVER_RETRY=1` | reset keeps the level (su15, tu93); tn36 read "The game is over." as stop; tr87 cleared L1 in 9 actions after its retry |
| level-up review | `DUCK_PROMPT_LEVELUP_REVIEW=1` | lf52 carried "reduce to 1" (really 2) and declared L2 impossible; ar25/s5i5 re-derived L1 late |
| MTP speculative decode (arm FM only) | `DUCK_VLLM_SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}` | a turn is 100 % decode; the snapshot ships `mtp.safetensors`; vLLM recipe for Qwen3.8-27B; vLLM >= 0.17 (we run 0.19) |
With all flags off the user prompt renders byte-identically to upstream (checked).

## Arms (25 public games, 1 pass, 28-way, 120 min/game — same recipe as qwen38-commit)
- **FM** = yield 300 + both prompt fixes + MTP. Launched first (one GPU slot free; the other
  holds `tycho-nooa-v12-sb26-duck-batch`).
- **F** = yield 300 + both prompt fixes, no MTP. Launched when a slot frees; attributes MTP.
- Baseline (recorded, not re-run): qwen38-commit, 22 levels, 14/25 games, 45 % yielded turns,
  median 192 s/turn.

## Mechanism checks (low noise; each is pass/fail on its own)
- F and FM: yielded-turn share **< 20 %** (from 45 %); ls20 takes >= 1 action.
- FM: vLLM log shows spec-decode acceptance, and median s/turn (or output tok/s per stream)
  **>= 25 % better than F**. Throughput alone is not accepted as proof the drafter works.
- FM startup failure (server does not come up) => relaunch as F, record MTP as not working here.

## Levels (sd of a 25-game delta ~ 4; noise floor from analysis/level-noise-floor.py)
| arm levels | vs 22 | reading |
|---|---|---|
| >= 30 | +8 | real gain at 2 sigma |
| 26–29 | +4…+7 | suggestive |
| 18–25 | -4…+3 | indistinguishable: keep a fix only on its mechanism check |
| <= 17 | <= -5 | suggestive of harm: find which fix before keeping any |
Also read: levels with the top game dropped (sb26 carried 4 in the baseline).

## What this does not decide
Submission. The user's order is fixes -> scaffolding (questioner v2) -> submission. The winning
arm here becomes the base the questioner is tested on.

## Cost
~2.3 h GPU per arm, 0 submission slots.

## AMENDMENT 2026-09-23 10:20Z, after FM, before F and Q are pushed
FM result in RESULT-FM.md (MTP saturates KV at 28-way; no throughput gain). Next, run together,
same day, both **without MTP**:
- **F** = yield 300 + game-over retry + level-up review.
- **Q** = F + questioner (`DUCK_CRITIC_ENABLED=1 DUCK_CRITIC_MODE=questioner
  DUCK_CRITIC_PERIOD_TURNS=2 DUCK_CRITIC_QUESTIONER_EFFORT=none`), offline grade in
  runs/critic-v2/grade/GRADE.md.
Q vs F is the paired comparison (same tree, same day, one flag). Levels table as above with F as
the reference (sd of delta ~4): >= +8 real, +4..+7 suggestive, -4..+3 indistinguishable, <= -5 harm.
Also report the delta with the top game dropped, and per-game deltas on sp80, tn36, sc25, dc22.
Q mechanism checks: >= 80 % of firings deliver a question; questioner time < 10 % of game wall;
0 host errors; share of delivered questions followed by an `Answer to observer:` line (uptake);
no question delivered twice (ledger). F vs FM tells whether the KV saturation came from MTP or
from the yield.

## AMENDMENT 2026-09-23 13:10Z — F2 vs Q3, two passes each (user: "yes do it")
Tree c19dee3 (questioner v3: plan-belief aim + form rotation; grade in runs/critic-v2/grade/GRADE.md).
- **F2** = F, `--n-passes 2`. **Q3** = F2 + questioner v3 (period 2, thinking off). No MTP.
  Run together; 50 episodes each, 2 waves, ~4.3 h.
- Levels: sd of a 50-episode delta ~ sqrt(2*50*0.35) ~ 5.9. >= +12 real, +6..+11 suggestive,
  -6..+5 indistinguishable, <= -7 harm. Also pooled with the first pair (Q vs F, -2): 75 episodes/arm.
- **Lock-break metric (primary for the questioner, many samples):** at each delivered question, compare
  the 10 actions after it with the level's actions before it; a *break* = the post-question actions
  touch an action/click target never used before in that level. Control: the same metric in F2 at
  matched points (every 2nd turn after the 2nd turn of a level). Questioner helps the mechanism if
  Q3's break rate exceeds F2's matched rate by >= 10 points.
- Mechanism checks as before: >= 80 % delivered, < 10 % of wall, 0 duplicates, answer-line rate.
